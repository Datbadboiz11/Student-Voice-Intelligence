from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import httpx
import pandas as pd
import streamlit as st


DEFAULT_API_URL = os.getenv("STUDENT_VOICE_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 180.0


def response_error(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    return str(detail or f"HTTP {response.status_code}")


def request_api(
    method: str,
    api_url: str,
    path: str,
    **kwargs: Any,
) -> tuple[httpx.Response | None, str | None]:
    try:
        response = httpx.request(
            method,
            f"{api_url.rstrip('/')}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except httpx.RequestError as exc:
        return None, f"Không kết nối được API: {exc}"

    if response.is_error:
        return None, response_error(response)
    return response, None


def render_single_prediction(api_url: str) -> None:
    st.subheader("Phân tích một feedback")
    with st.form("single-prediction"):
        text = st.text_area(
            "Nội dung feedback",
            placeholder="Ví dụ: Wifi phòng học quá yếu, máy chiếu bị mờ nên rất khó học.",
            height=150,
        )
        submitted = st.form_submit_button("Phân tích")

    if not submitted:
        return
    if not text.strip():
        st.warning("Hãy nhập nội dung feedback.")
        return

    with st.spinner("Đang phân tích feedback..."):
        response, error = request_api("POST", api_url, "/predict", json={"text": text})

    if error:
        st.error(error)
        return

    result = response.json()
    sentiment, topic, toxic, urgency = st.columns(4)
    sentiment.metric("Sentiment", result["sentiment"], f"{result['sentiment_confidence']:.1%}")
    topic.metric("Topic", result["topic"], f"{result['topic_confidence']:.1%}")
    toxic.metric("Toxic", "Có" if result["toxic"] else "Không")
    urgency.metric("Urgency", result["urgency"])

    with st.expander("Xem xác suất và JSON đầy đủ"):
        st.json(result)


def distribution_chart(rows: list[dict[str, Any]], title: str, label_column: str = "label") -> None:
    st.subheader(title)
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("Không có dữ liệu phù hợp.")
        return
    st.bar_chart(frame.set_index(label_column)["count"])


def get_analytics(api_url: str, params: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    response, error = request_api("GET", api_url, "/analytics", params=params or {})
    if error:
        return None, error
    return response.json(), None


def render_overview(api_url: str) -> None:
    st.subheader("Tổng quan dữ liệu")
    data, error = get_analytics(api_url)
    if error:
        st.error(error)
        return

    sentiment = {row["label"]: row["percentage"] for row in data["sentiment_distribution"]}
    urgency = {row["label"]: row["percentage"] for row in data["urgency_distribution"]}
    toxic = {row["label"]: row["percentage"] for row in data["toxic_distribution"]}
    metrics = st.columns(4)
    metrics[0].metric("Tổng feedback", f"{data['total_feedback']:,}")
    metrics[1].metric("Negative", f"{sentiment.get('negative', 0):.1f}%")
    metrics[2].metric("High urgency", f"{urgency.get('high', 0):.1f}%")
    metrics[3].metric("Toxic", f"{toxic.get(1, 0):.1f}%")

    left, right = st.columns(2)
    with left:
        distribution_chart(data["source_distribution"], "Feedback theo dataset")
        distribution_chart(data["topic_distribution"], "Phân bố chủ đề")
    with right:
        distribution_chart(data["sentiment_distribution"], "Phân bố sentiment")
        distribution_chart(data["urgency_distribution"], "Phân bố urgency")


def render_data_analytics(api_url: str) -> None:
    st.subheader("Phân tích dữ liệu")
    with st.form("analytics-filters"):
        filters = st.columns(5)
        dataset = filters[0].selectbox("Dataset", ["Tất cả", "NEU_ESC", "UIT_VSFC"])
        topic = filters[1].selectbox(
            "Chủ đề",
            ["Tất cả", "facilities", "teaching_learning", "student_services", "career_jobs", "events_news", "personal_social", "spam", "others"],
        )
        sentiment = filters[2].selectbox("Sentiment", ["Tất cả", "negative", "neutral", "positive"])
        urgency = filters[3].selectbox("Urgency", ["Tất cả", "high", "medium", "low"])
        toxic = filters[4].selectbox("Toxic", ["Tất cả", "Không", "Có"])
        submitted = st.form_submit_button("Áp dụng filter")

    params = st.session_state.get("analytics_params", {})
    if submitted:
        params = {}
        if dataset != "Tất cả":
            params["dataset"] = dataset
        if topic != "Tất cả":
            params["topic"] = topic
        if sentiment != "Tất cả":
            params["sentiment"] = sentiment
        if urgency != "Tất cả":
            params["urgency"] = urgency
        if toxic != "Tất cả":
            params["toxic"] = 1 if toxic == "Có" else 0
        st.session_state["analytics_params"] = params

    data, error = get_analytics(api_url, params)
    if error:
        st.error(error)
        return

    st.caption(f"Hiển thị {data['filtered_feedback']:,} / {data['total_feedback']:,} feedback")
    left, right = st.columns(2)
    with left:
        distribution_chart(data["negative_by_topic"], "Feedback negative theo chủ đề", "topic")
    with right:
        distribution_chart(data["topic_distribution"], "Phân bố chủ đề sau filter")

    st.subheader("Urgency theo chủ đề")
    urgency_frame = pd.DataFrame(data["urgency_by_topic"])
    if urgency_frame.empty:
        st.info("Không có dữ liệu phù hợp.")
    else:
        st.bar_chart(urgency_frame.pivot(index="topic", columns="urgency", values="count").fillna(0))

    st.subheader("Sentiment × Topic")
    matrix = pd.DataFrame(data["sentiment_topic_matrix"])
    if matrix.empty:
        st.info("Không có dữ liệu phù hợp.")
    else:
        matrix = matrix.pivot(index="topic", columns="sentiment", values="count").fillna(0).astype(int)
        st.dataframe(matrix, use_container_width=True)


def render_csv_prediction(api_url: str) -> None:
    st.subheader("Phân tích CSV hàng loạt")
    st.caption("CSV UTF-8 cần có cột bắt buộc: text. Tối đa 5.000 dòng.")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded_file is None:
        return

    if st.button("Phân tích CSV", type="primary"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv",
            )
        }
        with st.spinner("Đang phân tích file. Lần đầu có thể mất một lúc để load model..."):
            response, error = request_api("POST", api_url, "/predict-csv", files=files)

        if error:
            st.error(error)
            return

        st.session_state["csv_result"] = response.content

    csv_result = st.session_state.get("csv_result")
    if not csv_result:
        return

    result_frame = pd.read_csv(BytesIO(csv_result), encoding="utf-8-sig")
    st.success(f"Đã phân tích {len(result_frame)} feedback.")
    st.dataframe(result_frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Tải CSV kết quả",
        data=csv_result,
        file_name="student_voice_predictions.csv",
        mime="text/csv",
    )


def render_semantic_search(api_url: str) -> None:
    st.subheader("Tìm feedback tương tự")
    st.caption("Tìm theo ngữ nghĩa trong feedback đã được index bằng Qdrant.")

    with st.form("semantic-search"):
        query = st.text_input(
            "Câu tìm kiếm",
            placeholder="Ví dụ: Wifi phòng học quá yếu",
        )
        top_k = st.number_input("Số kết quả", min_value=1, max_value=20, value=5)
        filter_columns = st.columns(4)
        topic = filter_columns[0].selectbox(
            "Chủ đề",
            [
                "Tất cả",
                "facilities",
                "teaching_learning",
                "student_services",
                "career_jobs",
                "events_news",
                "personal_social",
                "spam",
                "others",
            ],
        )
        sentiment = filter_columns[1].selectbox(
            "Sentiment",
            ["Tất cả", "negative", "neutral", "positive"],
        )
        urgency = filter_columns[2].selectbox(
            "Urgency",
            ["Tất cả", "high", "medium", "low"],
        )
        toxic = filter_columns[3].selectbox(
            "Toxic",
            ["Tất cả", "Không", "Có"],
        )
        submitted = st.form_submit_button("Tìm kiếm")

    if not submitted:
        return
    if not query.strip():
        st.warning("Hãy nhập câu tìm kiếm.")
        return

    with st.spinner("Đang tìm các feedback gần nghĩa..."):
        payload = {"query": query, "top_k": int(top_k)}
        if topic != "Tất cả":
            payload["topic"] = topic
        if sentiment != "Tất cả":
            payload["sentiment"] = sentiment
        if urgency != "Tất cả":
            payload["urgency"] = urgency
        if toxic != "Tất cả":
            payload["toxic"] = 1 if toxic == "Có" else 0
        response, error = request_api(
            "POST",
            api_url,
            "/search",
            json=payload,
        )

    if error:
        st.error(error)
        return

    results = response.json()["results"]
    if not results:
        st.info("Không tìm thấy feedback phù hợp.")
        return

    frame = pd.DataFrame(results)
    frame["rerank_score"] = frame["rerank_score"].map(lambda value: f"{value:.3f}")
    frame["vector_score"] = frame["vector_score"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        frame[
            [
                "rerank_score",
                "vector_score",
                "text",
                "topic",
                "sentiment",
                "urgency",
                "toxic",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_rag_chatbot(api_url: str) -> None:
    st.subheader("Hỏi đáp từ feedback sinh viên")
    st.caption("Câu trả lời được giới hạn trong các feedback đã truy xuất và có kèm bằng chứng.")

    with st.form("rag-chatbot"):
        question = st.text_area(
            "Câu hỏi",
            placeholder="Ví dụ: Sinh viên đang phàn nàn gì về Wifi và phòng học?",
            height=100,
        )
        top_k = st.number_input("Số feedback làm bằng chứng", min_value=1, max_value=10, value=6)
        filter_columns = st.columns(4)
        topic = filter_columns[0].selectbox(
            "Chủ đề",
            [
                "Tất cả",
                "facilities",
                "teaching_learning",
                "student_services",
                "career_jobs",
                "events_news",
                "personal_social",
                "spam",
                "others",
            ],
            key="rag-topic",
        )
        sentiment = filter_columns[1].selectbox(
            "Sentiment",
            ["Tất cả", "negative", "neutral", "positive"],
            key="rag-sentiment",
        )
        urgency = filter_columns[2].selectbox(
            "Urgency",
            ["Tất cả", "high", "medium", "low"],
            key="rag-urgency",
        )
        toxic = filter_columns[3].selectbox(
            "Toxic",
            ["Tất cả", "Không", "Có"],
            key="rag-toxic",
        )
        submitted = st.form_submit_button("Hỏi dữ liệu")

    if not submitted:
        return
    if not question.strip():
        st.warning("Hãy nhập câu hỏi.")
        return

    with st.spinner("Đang truy xuất feedback và tạo câu trả lời..."):
        payload = {"question": question, "top_k": int(top_k)}
        if topic != "Tất cả":
            payload["topic"] = topic
        if sentiment != "Tất cả":
            payload["sentiment"] = sentiment
        if urgency != "Tất cả":
            payload["urgency"] = urgency
        if toxic != "Tất cả":
            payload["toxic"] = 1 if toxic == "Có" else 0
        response, error = request_api(
            "POST",
            api_url,
            "/ask",
            json=payload,
        )

    if error:
        st.error(error)
        return

    result = response.json()
    st.markdown(result["answer"])
    if not result["evidence"]:
        st.info("Không có feedback phù hợp để làm bằng chứng.")
        return

    st.caption(f"Dùng {result['retrieved_count']} feedback làm bằng chứng")
    for row in result["evidence"]:
        title = f"[{row['rank']}] {row.get('topic') or 'unknown'} | rerank {row.get('rerank_score', 0):.3f}"
        with st.expander(title):
            st.write(row.get("text", ""))
            st.json(
                {
                    key: row.get(key)
                    for key in ("source_dataset", "sentiment", "urgency", "toxic", "vector_score")
                }
            )


def main() -> None:
    st.set_page_config(page_title="Student Voice Intelligence", page_icon="🎓", layout="wide")
    st.title("Student Voice Intelligence")
    st.caption("Phân tích feedback sinh viên tiếng Việt bằng FastAPI và PhoBERT.")

    with st.sidebar:
        st.header("Kết nối API")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        if st.button("Kiểm tra API"):
            response, error = request_api("GET", api_url, "/health")
            if error:
                st.error(error)
            else:
                health = response.json()
                if all(value for key, value in health.items() if key.endswith("_exists")):
                    st.success("API và model đã sẵn sàng.")
                else:
                    st.warning("API đang chạy nhưng thiếu một hoặc nhiều model.")
                st.json(health)

    overview_tab, single_tab, csv_tab, analytics_tab, search_tab, rag_tab = st.tabs(
        ["Tổng quan", "Một feedback", "Upload CSV", "Phân tích dữ liệu", "Tìm kiếm ngữ nghĩa", "RAG Chatbot"]
    )
    with overview_tab:
        render_overview(api_url)
    with single_tab:
        render_single_prediction(api_url)
    with csv_tab:
        render_csv_prediction(api_url)
    with analytics_tab:
        render_data_analytics(api_url)
    with search_tab:
        render_semantic_search(api_url)
    with rag_tab:
        render_rag_chatbot(api_url)


if __name__ == "__main__":
    main()
