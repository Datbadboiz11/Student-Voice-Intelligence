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

    single_tab, csv_tab = st.tabs(["Một feedback", "Upload CSV"])
    with single_tab:
        render_single_prediction(api_url)
    with csv_tab:
        render_csv_prediction(api_url)


if __name__ == "__main__":
    main()
