"use client";

import { FormEvent, ReactNode, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Bot,
  CheckCircle2,
  ChevronRight,
  Database,
  FileUp,
  GraduationCap,
  LayoutDashboard,
  LoaderCircle,
  MessageSquareText,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
  Upload,
  Wifi
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { ReportWorkbench, ReviewWorkbench, TopicDiscoveryWorkbench } from "./v3-workbench";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Distribution = { label: string | number; count: number; percentage: number };
type TopicCount = { topic: string; count: number };
type GroupCount = { topic: string; urgency?: string; sentiment?: string; count: number };
type Analytics = {
  total_feedback: number;
  filtered_feedback: number;
  source_distribution: Distribution[];
  sentiment_distribution: Distribution[];
  topic_distribution: Distribution[];
  urgency_distribution: Distribution[];
  toxic_distribution: Distribution[];
  negative_by_topic: TopicCount[];
  urgency_by_topic: GroupCount[];
  sentiment_topic_matrix: GroupCount[];
};
type SearchResult = {
  id: string;
  text: string;
  topic?: string;
  sentiment?: string;
  urgency?: string;
  toxic?: number;
  vector_score: number;
  rerank_score: number;
};
type RAGResult = {
  answer: string;
  grounded: boolean;
  retrieved_count: number;
  evidence: Array<SearchResult & { rank: number; source_dataset?: string }>;
};
type ChatMessage = { role: "user" | "assistant"; content: string; result?: RAGResult | null };
type ChatSession = { id: number; title: string; created_at: string; updated_at: string };
type ChatSessionDetail = ChatSession & { messages: ChatMessage[] };
type ChatAskResponse = { session: ChatSession; user_message: ChatMessage; assistant_message: ChatMessage };
type Prediction = {
  sentiment: string;
  sentiment_confidence: number;
  topic: string;
  topic_confidence: number;
  toxic: number;
  urgency: string;
};
type View = "overview" | "analytics" | "predict" | "csv" | "search" | "rag" | "review" | "reports" | "discovery";

const TOPICS = ["facilities", "teaching_learning", "student_services", "career_jobs", "events_news", "personal_social", "spam", "others"];
const SENTIMENTS = ["negative", "neutral", "positive"];
const URGENCIES = ["high", "medium", "low"];

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function percent(rows: Distribution[], label: string | number) {
  return rows.find((row) => String(row.label) === String(label))?.percentage || 0;
}

function count(rows: Distribution[], label: string | number) {
  return rows.find((row) => String(row.label) === String(label))?.count || 0;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("vi-VN").format(value);
}

function apiFilter(query: Record<string, string | number | undefined>) {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value));
  });
  const value = params.toString();
  return value ? `?${value}` : "";
}

export function Dashboard() {
  const [view, setView] = useState<View>("overview");
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const [apiReady, setApiReady] = useState(false);

  const loadAnalytics = async (filters: Record<string, string | number | undefined> = {}) => {
    try {
      setAnalyticsError(null);
      const data = await apiRequest<Analytics>(`/analytics${apiFilter(filters)}`);
      setAnalytics(data);
    } catch (error) {
      setAnalyticsError(error instanceof Error ? error.message : "Không thể tải dữ liệu analytics.");
    }
  };

  useEffect(() => {
    loadAnalytics();
    apiRequest<{ status: string }>("/health")
      .then(() => setApiReady(true))
      .catch(() => setApiReady(false));
  }, []);

  const navigation: Array<{ id: View; label: string; icon: typeof LayoutDashboard }> = [
    { id: "overview", label: "Tổng quan", icon: LayoutDashboard },
    { id: "analytics", label: "Phân tích dữ liệu", icon: BarChart3 },
    { id: "predict", label: "Dự đoán feedback", icon: Sparkles },
    { id: "csv", label: "Phân tích CSV", icon: FileUp },
    { id: "search", label: "Semantic Search", icon: Search },
    { id: "rag", label: "RAG Chatbot", icon: Bot },
    { id: "review", label: "Duyệt urgency", icon: CheckCircle2 },
    { id: "reports", label: "Báo cáo", icon: MessageSquareText },
    { id: "discovery", label: "Khám phá chủ đề", icon: Activity }
  ];

  return (
    <main className="min-h-screen bg-[#fcfbf7] text-[#142033] lg:flex">
      <aside className="flex flex-col bg-[#071d35] px-3 py-6 text-slate-100 lg:fixed lg:inset-y-0 lg:w-64">
        <div className="mb-10 flex items-center gap-3 px-3">
          <div className="grid h-11 w-11 place-items-center rounded-md border border-white/30 text-white">
            <GraduationCap size={25} />
          </div>
          <div>
            <p className="text-base font-bold leading-tight">Student Voice</p>
            <p className="text-xs uppercase tracking-[0.14em] text-slate-300">Intelligence</p>
          </div>
        </div>

        <nav className="grid gap-1 sm:grid-cols-2 lg:block">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-sm font-medium transition ${
                view === id ? "bg-[#315de0] text-white" : "text-slate-200 hover:bg-white/10"
              }`}
            >
              <Icon size={19} />
              {label}
            </button>
          ))}
        </nav>

        <div className="mt-auto border-t border-white/20 px-3 pt-6 text-xs text-slate-300">
          <span className={`h-2.5 w-2.5 rounded-full ${apiReady ? "bg-emerald-400" : "bg-rose-400"}`} />
          {apiReady ? "API sẵn sàng" : "Đang kiểm tra API"}
        </div>
      </aside>

      <section className="min-w-0 flex-1 lg:ml-64">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-[#d9d1c4] bg-[#fcfbf7]/95 px-5 py-4 backdrop-blur lg:px-10">
          <div>
            <p className="text-xl font-bold text-[#142033]">Student Voice Intelligence</p>
            <h1 className="mt-0.5 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{navigation.find((item) => item.id === view)?.label}</h1>
          </div>
          <div className={`rounded-md border px-3 py-1.5 text-xs font-semibold ${apiReady ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-rose-300 bg-rose-50 text-rose-700"}`}>
            {apiReady ? "● API sẵn sàng" : "● API chưa sẵn sàng"}
          </div>
        </header>

        <div className="mx-auto max-w-[1600px] p-5 lg:p-8">
          {analyticsError && <ErrorNotice message={analyticsError} />}
          {view === "overview" && <Overview data={analytics} onNavigate={setView} />}
          {view === "analytics" && <AnalyticsView data={analytics} onApply={loadAnalytics} />}
          {view === "predict" && <PredictionView />}
          {view === "csv" && <CsvView />}
          {view === "search" && <SearchView />}
          {view === "rag" && <RagSessionView />}
          {view === "review" && <ReviewWorkbench />}
          {view === "reports" && <ReportWorkbench />}
          {view === "discovery" && <TopicDiscoveryWorkbench />}
        </div>
      </section>
    </main>
  );
}

function Overview({ data, onNavigate }: { data: Analytics | null; onNavigate: (view: View) => void }) {
  if (!data) return <LoadingCard label="Đang tải tổng quan dữ liệu..." />;
  const totalNegative = count(data.sentiment_distribution, "negative");
  const totalUrgency = data.urgency_distribution.reduce((sum, row) => sum + row.count, 0) || 1;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#d9d1c4] pb-4">
        <div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#315de0]">Báo cáo dữ liệu</p><h2 className="mt-1 text-3xl font-bold tracking-tight">Điểm nóng hiện tại</h2></div>
        <p className="max-w-md text-sm leading-6 text-slate-500">Tổng hợp vấn đề sinh viên phản ánh nhiều nhất từ dữ liệu đã enriched.</p>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.35fr_.85fr]">
        <section className="border border-[#d9d1c4] bg-[#fffefa] p-5">
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[#ded7cd] pb-4">
            <div><h3 className="text-2xl font-bold">Chủ đề cần chú ý</h3><p className="mt-1 text-sm text-slate-500">Tỷ trọng trong toàn bộ feedback tiêu cực</p></div>
            <div className="text-right"><p className="text-xs font-semibold uppercase tracking-[0.12em] text-[#c94e3d]">Tỷ lệ phản hồi tiêu cực</p><p className="mt-1 text-5xl font-semibold text-[#c94e3d]">{percent(data.sentiment_distribution, "negative").toFixed(1)}%</p></div>
          </div>
          <div className="mt-5 space-y-4">{data.negative_by_topic.slice(0, 5).map((row, index) => <EditorialTopicRow key={row.topic} row={row} total={totalNegative} index={index} />)}</div>
          <button onClick={() => onNavigate("analytics")} className="mt-5 inline-flex items-center gap-1 text-sm font-semibold text-[#315de0] hover:underline">Xem phân tích đầy đủ <ChevronRight size={16} /></button>
        </section>
        <section className="space-y-4">
          <div className="border border-[#d9d1c4] bg-[#fffefa] p-5"><h3 className="text-xl font-bold">Tổng quan dữ liệu</h3><div className="mt-4 grid grid-cols-2 divide-x divide-y divide-[#ded7cd] border-y border-[#ded7cd]">{[{ label: "Tổng feedback", value: data.total_feedback, color: "text-[#315de0]" }, { label: "Tiêu cực", value: totalNegative, color: "text-[#c94e3d]" }, { label: "Tích cực", value: count(data.sentiment_distribution, "positive"), color: "text-[#4e8069]" }, { label: "Trung lập", value: count(data.sentiment_distribution, "neutral"), color: "text-[#b0813e]" }].map((item) => <div className="p-4" key={item.label}><p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{item.label}</p><p className={`mt-2 text-3xl font-semibold ${item.color}`}>{formatNumber(item.value)}</p></div>)}</div></div>
          <section className="border border-[#d9d1c4] bg-[#fffefa] p-5"><h3 className="text-xl font-bold">Mức độ khẩn cấp</h3><p className="mt-1 text-sm text-slate-500">Phân bố urgency trong dữ liệu</p><div className="mt-6 flex h-3 overflow-hidden rounded-full bg-slate-100">{data.urgency_distribution.map((row) => <div key={String(row.label)} className={row.label === "high" ? "bg-[#c94e3d]" : row.label === "medium" ? "bg-[#c79040]" : "bg-[#4e8069]"} style={{ width: `${(row.count / totalUrgency) * 100}%` }} />)}</div><div className="mt-4 grid grid-cols-3 gap-2">{data.urgency_distribution.map((row) => <div key={String(row.label)}><p className="text-xs font-semibold text-slate-500">{row.label}</p><p className="mt-1 text-xl font-semibold">{formatNumber(row.count)}</p></div>)}</div></section>
        </section>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.35fr_.85fr]">
        <section className="border border-[#d9d1c4] bg-[#fffefa] p-5"><h3 className="text-xl font-bold">Chủ đề nổi bật</h3><p className="mt-1 text-sm text-slate-500">Số feedback tiêu cực theo chủ đề</p><EditorialTable rows={data.negative_by_topic} total={totalNegative} /></section>
        <section className="border border-[#d9d1c4] bg-[#fffefa] p-5"><h3 className="text-xl font-bold">Câu hỏi gợi ý cho RAG</h3><p className="mt-1 text-sm text-slate-500">Khai thác sâu hơn từ feedback gốc</p><div className="mt-4 divide-y divide-[#ded7cd]">{["Vấn đề nào về cơ sở vật chất được phản ánh nhiều nhất?", "Sinh viên đề xuất gì để cải thiện việc học?", "Những chủ đề nào có feedback negative cao?", "Có vấn đề nào cần xử lý khẩn cấp không?"].map((question) => <button key={question} onClick={() => onNavigate("rag")} className="flex w-full items-center gap-3 py-3 text-left text-sm text-slate-700 hover:text-[#315de0]"><MessageSquareText size={16} className="text-[#315de0]" /><span>{question}</span><ChevronRight size={16} className="ml-auto" /></button>)}</div><button onClick={() => onNavigate("rag")} className="mt-4 text-sm font-semibold text-[#315de0] hover:underline">Mở RAG Chatbot <ChevronRight className="inline" size={15} /></button></section>
      </div>
    </div>
  );
}

function AnalyticsView({ data, onApply }: { data: Analytics | null; onApply: (filters: Record<string, string | number | undefined>) => void }) {
  const [filters, setFilters] = useState<Record<string, string>>({ dataset: "", topic: "", sentiment: "", urgency: "", toxic: "" });
  const apply = (event: FormEvent) => {
    event.preventDefault();
    onApply({ ...filters, toxic: filters.toxic === "" ? undefined : Number(filters.toxic) });
  };
  return (
    <div className="space-y-6">
      <div><p className="text-sm font-semibold uppercase tracking-[0.18em] text-blue-600">Khám phá dữ liệu</p><h2 className="mt-2 text-3xl font-bold tracking-tight">Phân tích phản hồi</h2></div>
      <Panel title="Bộ lọc dữ liệu" subtitle="Áp dụng filter cho toàn bộ biểu đồ bên dưới">
        <form onSubmit={apply} className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Select value={filters.dataset} onChange={(value) => setFilters({ ...filters, dataset: value })} label="Dataset" options={["", "NEU_ESC", "UIT_VSFC"]} />
          <Select value={filters.topic} onChange={(value) => setFilters({ ...filters, topic: value })} label="Chủ đề" options={["", ...TOPICS]} />
          <Select value={filters.sentiment} onChange={(value) => setFilters({ ...filters, sentiment: value })} label="Sentiment" options={["", ...SENTIMENTS]} />
          <Select value={filters.urgency} onChange={(value) => setFilters({ ...filters, urgency: value })} label="Urgency" options={["", ...URGENCIES]} />
          <Select value={filters.toxic} onChange={(value) => setFilters({ ...filters, toxic: value })} label="Toxic" options={["", "0", "1"]} />
          <button className="mt-5 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700">Áp dụng filter</button>
        </form>
      </Panel>
      {!data ? <LoadingCard label="Đang tải analytics..." /> : <>
        <p className="text-sm text-slate-500">Hiển thị <strong className="text-slate-900">{formatNumber(data.filtered_feedback)}</strong> / {formatNumber(data.total_feedback)} feedback</p>
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title="Feedback negative theo chủ đề"><TopicBars rows={data.negative_by_topic} /></Panel>
          <Panel title="Phân bố chủ đề sau filter"><BarList rows={data.topic_distribution} colors={["bg-blue-600"]} /></Panel>
        </div>
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title="Urgency theo chủ đề"><GroupedBars rows={data.urgency_by_topic} groupKey="urgency" /></Panel>
          <Panel title="Sentiment × Topic"><Matrix rows={data.sentiment_topic_matrix} /></Panel>
        </div>
      </>}
    </div>
  );
}

function PredictionView() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!text.trim()) return setError("Hãy nhập nội dung feedback.");
    setLoading(true); setError(null);
    try { setResult(await apiRequest<Prediction>("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể phân tích feedback."); }
    finally { setLoading(false); }
  };
  return <div className="grid gap-6 xl:grid-cols-[1.1fr_.9fr]"><Panel title="Dự đoán một feedback" subtitle="PhoBERT phân loại sentiment và topic; baseline xác định toxic và urgency"><form onSubmit={submit} className="space-y-4"><textarea value={text} onChange={(event) => setText(event.target.value)} className="min-h-52 w-full rounded-xl border border-slate-200 bg-slate-50 p-4 outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100" placeholder="Ví dụ: Wi-Fi phòng học quá yếu, máy chiếu bị mờ nên rất khó học." /><button className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700" disabled={loading}><Sparkles size={17} />{loading ? "Đang phân tích..." : "Phân tích feedback"}</button>{error && <ErrorNotice message={error} />}</form></Panel><Panel title="Kết quả phân tích" subtitle="Sẽ hiển thị sau khi gửi feedback">{result ? <PredictionResult result={result} /> : <Empty text="Chưa có kết quả phân tích." />}</Panel></div>;
}

function CsvView() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [download, setDownload] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) return setError("Hãy chọn file CSV.");
    setLoading(true); setError(null); setDownload(null);
    try {
      const payload = new FormData(); payload.append("file", file);
      const response = await fetch(`${API_URL}/predict-csv`, { method: "POST", body: payload });
      if (!response.ok) { const body = await response.json().catch(() => null); throw new Error(body?.detail || `HTTP ${response.status}`); }
      setDownload(URL.createObjectURL(await response.blob()));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể phân tích CSV."); }
    finally { setLoading(false); }
  };
  return <Panel title="Phân tích CSV hàng loạt" subtitle="CSV UTF-8 cần có cột bắt buộc text; tối đa 5.000 dòng."><form onSubmit={submit} className="space-y-5"><label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center hover:border-blue-400 hover:bg-blue-50"><Upload className="mb-3 text-blue-600" size={30} /><span className="font-semibold">{file ? file.name : "Chọn file CSV hoặc kéo thả vào đây"}</span><span className="mt-1 text-sm text-slate-500">Chỉ nhận định dạng .csv</span><input className="hidden" type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0] || null)} /></label><button className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700" disabled={loading}>{loading ? "Đang xử lý..." : "Phân tích CSV"}</button>{download && <a className="ml-3 rounded-xl border border-blue-200 bg-blue-50 px-5 py-3 text-sm font-semibold text-blue-700" href={download} download="student_voice_predictions.csv">Tải file kết quả</a>}{error && <ErrorNotice message={error} />}</form></Panel>;
}

function SearchView() {
  const [query, setQuery] = useState(""); const [topic, setTopic] = useState(""); const [results, setResults] = useState<SearchResult[]>([]); const [error, setError] = useState<string | null>(null); const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent) => { event.preventDefault(); if (!query.trim()) return setError("Hãy nhập câu tìm kiếm."); setLoading(true); setError(null); try { const body = await apiRequest<{ results: SearchResult[] }>("/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, top_k: 6, ...(topic ? { topic } : {}) }) }); setResults(body.results); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tìm kiếm."); } finally { setLoading(false); } };
  return <div className="space-y-6"><Panel title="Semantic Search" subtitle="Tìm feedback liên quan bằng embedding, Qdrant và CrossEncoder reranking"><form onSubmit={submit} className="grid gap-3 md:grid-cols-[1fr_220px_auto]"><input value={query} onChange={(event) => setQuery(event.target.value)} className="rounded-xl border border-slate-200 px-4 py-3 outline-none focus:border-blue-500" placeholder="Ví dụ: Wi-Fi phòng học quá yếu" /><Select value={topic} onChange={setTopic} label="" options={["", ...TOPICS]} /><button className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700">{loading ? "Đang tìm..." : "Tìm kiếm"}</button></form>{error && <ErrorNotice message={error} />}</Panel><ResultList results={results} /></div>;
}

function RagSessionView() {
  const [question, setQuestion] = useState("");
  const [topic, setTopic] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadSessions = async () => {
    const response = await apiRequest<{ items: ChatSession[] }>("/chat-sessions");
    setSessions(response.items);
  };

  useEffect(() => {
    loadSessions().catch((reason) => setError(reason instanceof Error ? reason.message : "Không thể tải lịch sử chat."));
  }, []);

  const createSession = async () => {
    const session = await apiRequest<ChatSession>("/chat-sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    setActiveSessionId(session.id);
    setMessages([]);
    setError(null);
    await loadSessions();
    return session.id;
  };

  const openSession = async (sessionId: number) => {
    try {
      setLoading(true);
      setError(null);
      const session = await apiRequest<ChatSessionDetail>(`/chat-sessions/${sessionId}`);
      setActiveSessionId(session.id);
      setMessages(session.messages);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể mở phiên trò chuyện.");
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async (sessionId: number) => {
    if (!window.confirm("Xóa phiên trò chuyện này?")) return;
    try {
      await apiRequest<{ deleted: boolean }>(`/chat-sessions/${sessionId}`, { method: "DELETE" });
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      await loadSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể xóa phiên trò chuyện.");
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const text = question.trim();
    if (!text) return setError("Hãy nhập câu hỏi.");
    setLoading(true);
    setError(null);
    try {
      const sessionId = activeSessionId || await createSession();
      setMessages((items) => [...items, { role: "user", content: text }]);
      setQuestion("");
      const response = await apiRequest<ChatAskResponse>(`/chat-sessions/${sessionId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, top_k: 6, ...(topic ? { topic } : {}), ...(sentiment ? { sentiment } : {}) }),
      });
      setMessages((items) => [...items, response.assistant_message]);
      await loadSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể tạo câu trả lời.");
    } finally {
      setLoading(false);
    }
  };

  return <div className="grid gap-6 xl:grid-cols-[260px_.9fr_1.1fr]">
    <Panel title="Phiên trò chuyện" subtitle="Lịch sử được lưu trong SQLite">
      <button type="button" onClick={() => { createSession().catch((reason) => setError(reason instanceof Error ? reason.message : "Không thể tạo phiên mới.")); }} className="mb-4 w-full rounded-xl bg-[#315de0] px-3 py-2.5 text-sm font-semibold text-white hover:bg-[#2549ba]">+ Phiên mới</button>
      <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1">
        {sessions.length ? sessions.map((session) => <div key={session.id} className={`group flex items-center gap-2 border p-2 ${activeSessionId === session.id ? "border-[#315de0] bg-blue-50" : "border-[#ded7cd] hover:bg-[#faf8f2]"}`}>
          <button type="button" onClick={() => { openSession(session.id); }} className="min-w-0 flex-1 text-left">
            <p className="truncate text-sm font-semibold text-slate-700">{session.title}</p>
            <p className="mt-1 text-xs text-slate-400">{new Date(`${session.updated_at}Z`).toLocaleDateString("vi-VN")}</p>
          </button>
          <button type="button" onClick={() => { deleteSession(session.id); }} aria-label={`Xóa ${session.title}`} className="p-1 text-slate-300 hover:text-rose-600"><Trash2 size={15} /></button>
        </div>) : <p className="px-1 text-sm leading-6 text-slate-500">Chưa có phiên nào được lưu.</p>}
      </div>
    </Panel>
    <Panel title="RAG Chatbot" subtitle={activeSessionId ? "Đang hỏi trong phiên hiện tại" : "Tạo phiên khi gửi câu hỏi đầu tiên"}>
      <form onSubmit={submit} className="space-y-4">
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} className="min-h-40 w-full rounded-xl border border-slate-200 bg-slate-50 p-4 outline-none focus:border-blue-500" placeholder="Hỏi tiếp, ví dụ: Vậy vấn đề đó xuất hiện ở những phòng nào?" />
        <div className="grid gap-3 sm:grid-cols-2"><Select value={topic} onChange={setTopic} label="Chủ đề" options={["", ...TOPICS]} /><Select value={sentiment} onChange={setSentiment} label="Sentiment" options={["", ...SENTIMENTS]} /></div>
        <button className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700" disabled={loading}><Send size={16} />{loading ? "Đang tổng hợp..." : "Gửi câu hỏi"}</button>
        {error && <ErrorNotice message={error} />}
      </form>
    </Panel>
    <Panel title="Cuộc trò chuyện" subtitle="Mỗi câu trả lời có evidence riêng">
      {messages.length ? <div className="space-y-4">
        {messages.map((message, index) => <div key={index} className={message.role === "user" ? "ml-8 border-l-2 border-[#315de0] bg-blue-50 p-4" : "mr-4 border border-[#d9d1c4] bg-[#fffefa] p-4"}>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{message.role === "user" ? "Bạn" : "Student Voice"}</p>
          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{message.content}</p>
          {message.result && <details className="mt-3"><summary className="cursor-pointer text-xs font-semibold text-[#315de0]">{message.result.retrieved_count} feedback làm bằng chứng</summary><div className="mt-3 space-y-2">{message.result.evidence.map((row) => <article key={`${index}-${row.id}-${row.rank}`} className="border border-slate-200 p-3 text-sm"><div className="mb-2 flex flex-wrap gap-2"><Badge text={`#${row.rank}`} tone="blue" /><Badge text={row.topic || "unknown"} tone="slate" />{typeof row.rerank_score === "number" && <Badge text={`rerank ${row.rerank_score.toFixed(3)}`} tone="teal" />}</div><p className="text-slate-600">{row.text}</p></article>)}</div></details>}
        </div>)}
        {loading && <div className="text-sm text-slate-500"><LoaderCircle className="mr-2 inline animate-spin" size={16} />Đang tìm evidence và tổng hợp...</div>}
      </div> : <Empty text="Tạo phiên mới hoặc chọn một phiên cũ để tiếp tục trò chuyện." />}
    </Panel>
  </div>;
}

function RagView() {
  const [question, setQuestion] = useState(""); const [topic, setTopic] = useState(""); const [sentiment, setSentiment] = useState(""); const [result, setResult] = useState<RAGResult | null>(null); const [error, setError] = useState<string | null>(null); const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent) => { event.preventDefault(); if (!question.trim()) return setError("Hãy nhập câu hỏi."); setLoading(true); setError(null); try { setResult(await apiRequest<RAGResult>("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, top_k: 6, ...(topic ? { topic } : {}), ...(sentiment ? { sentiment } : {}) }) })); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tạo câu trả lời."); } finally { setLoading(false); } };
  return <div className="grid gap-6 xl:grid-cols-[.9fr_1.1fr]"><Panel title="RAG Chatbot" subtitle="Gemini chỉ trả lời dựa trên các feedback đã truy xuất"><form onSubmit={submit} className="space-y-4"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} className="min-h-40 w-full rounded-xl border border-slate-200 bg-slate-50 p-4 outline-none focus:border-blue-500" placeholder="Sinh viên đang phàn nàn gì về Wi-Fi và phòng học?" /><div className="grid gap-3 sm:grid-cols-2"><Select value={topic} onChange={setTopic} label="Chủ đề" options={["", ...TOPICS]} /><Select value={sentiment} onChange={setSentiment} label="Sentiment" options={["", ...SENTIMENTS]} /></div><button className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700" disabled={loading}><Send size={16} />{loading ? "Đang tổng hợp..." : "Hỏi dữ liệu"}</button>{error && <ErrorNotice message={error} />}</form></Panel><Panel title="Câu trả lời có căn cứ" subtitle="Evidence hiển thị ngay bên dưới câu trả lời">{result ? <div className="space-y-5"><div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-5 text-sm leading-7 text-slate-700 whitespace-pre-wrap">{result.answer}</div><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{result.retrieved_count} feedback làm bằng chứng</p>{result.evidence.map((row) => <article key={row.id} className="rounded-xl border border-slate-200 p-4"><div className="mb-2 flex flex-wrap gap-2"><Badge text={`#${row.rank}`} tone="blue" /><Badge text={row.topic || "unknown"} tone="slate" /><Badge text={`rerank ${row.rerank_score.toFixed(3)}`} tone="teal" /></div><p className="text-sm leading-6 text-slate-700">{row.text}</p></article>)}</div> : <Empty text="Đặt câu hỏi để Gemini tổng hợp feedback có citation." />}</Panel></div>;
}

function Panel({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) { return <section className="border border-[#d9d1c4] bg-[#fffefa] p-5"><div className="mb-5 border-b border-[#ded7cd] pb-4"><h3 className="text-xl font-bold text-[#142033]">{title}</h3>{subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}</div>{children}</section>; }
function LoadingCard({ label }: { label: string }) { return <div className="flex min-h-60 items-center justify-center border border-[#d9d1c4] bg-[#fffefa] text-sm text-slate-500"><LoaderCircle className="mr-2 animate-spin" size={18} />{label}</div>; }
function Empty({ text }: { text: string }) { return <div className="grid min-h-44 place-items-center border border-dashed border-[#d9d1c4] bg-[#faf8f2] px-6 text-center text-sm text-slate-500">{text}</div>; }
function ErrorNotice({ message }: { message: string }) { return <div className="mt-4 flex gap-2 border border-rose-300 bg-rose-50 p-3 text-sm text-rose-700"><AlertCircle size={18} className="shrink-0" />{message}</div>; }
function Badge({ text, tone }: { text: string; tone: "blue" | "teal" | "slate" }) { const colors = { blue: "bg-blue-50 text-blue-700", teal: "bg-emerald-50 text-emerald-700", slate: "bg-slate-100 text-slate-600" }; return <span className={`rounded-md px-2 py-1 text-xs font-semibold ${colors[tone]}`}>{text}</span>; }
function Select({ value, onChange, label, options }: { value: string; onChange: (value: string) => void; label: string; options: string[] }) { return <label className="block text-sm font-medium text-slate-600">{label && <span className="mb-1.5 block">{label}</span>}<select value={value} onChange={(event) => onChange(event.target.value)} className="w-full rounded-sm border border-[#d9d1c4] bg-[#fffefa] px-3 py-2.5 text-sm outline-none focus:border-[#315de0]">{options.map((option) => <option key={option || "all"} value={option}>{option === "" ? "Tất cả" : option === "0" ? "Không" : option === "1" ? "Có" : option}</option>)}</select></label>; }

function EditorialTopicRow({ row, total, index }: { row: TopicCount; total: number; index: number }) {
  const colors = ["bg-[#c94e3d]", "bg-[#315de0]", "bg-[#4e8069]", "bg-[#b0813e]", "bg-[#5b6f8a]"];
  const percentage = total ? (row.count / total) * 100 : 0;
  return <div className="grid grid-cols-[155px_1fr_58px] items-center gap-3 text-sm"><div className="flex items-center gap-2"><span className={`grid h-6 w-6 place-items-center rounded-full text-xs font-bold text-white ${colors[index % colors.length]}`}>{index + 1}</span><span className="truncate font-medium">{row.topic}</span></div><div className="h-3 bg-[#efe9df]"><div className={`h-full ${colors[index % colors.length]}`} style={{ width: `${percentage}%` }} /></div><span className="text-right text-base font-semibold">{percentage.toFixed(1)}%</span></div>;
}

function EditorialTable({ rows, total }: { rows: TopicCount[]; total: number }) {
  return <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm"><thead><tr className="border-y border-[#ded7cd] text-[11px] uppercase tracking-wide text-slate-500"><th className="py-3 font-semibold">#</th><th className="py-3 font-semibold">Chủ đề</th><th className="py-3 text-right font-semibold">Feedback negative</th><th className="py-3 text-right font-semibold">Tỷ trọng</th><th className="py-3 text-right font-semibold">Ưu tiên</th></tr></thead><tbody>{rows.slice(0, 6).map((row, index) => { const share = total ? (row.count / total) * 100 : 0; return <tr className="border-b border-[#ece6dc]" key={row.topic}><td className="py-3 text-slate-500">{index + 1}</td><td className="py-3 font-medium">{row.topic}</td><td className="py-3 text-right">{formatNumber(row.count)}</td><td className="py-3 text-right text-[#c94e3d]">{share.toFixed(1)}%</td><td className="py-3 text-right"><span className="border border-[#c8a36c] px-2 py-1 text-xs text-[#9a6722]">{index < 2 ? "Cần theo dõi" : "Trung bình"}</span></td></tr>; })}</tbody></table></div>;
}

function MetricCard({ label, value, helper, tone, icon: Icon }: { label: string; value: string; helper: string; tone: "blue" | "red" | "amber" | "violet"; icon: LucideIcon }) { const tones = { blue: "bg-blue-50 text-blue-600", red: "bg-rose-50 text-rose-600", amber: "bg-amber-50 text-amber-600", violet: "bg-violet-50 text-violet-600" }; return <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card"><div className="flex items-start justify-between"><div><p className="text-sm font-medium text-slate-500">{label}</p><p className="mt-2 text-3xl font-bold tracking-tight">{value}</p></div><div className={`rounded-xl p-3 ${tones[tone]}`}><Icon size={22} /></div></div><p className="mt-4 text-xs text-slate-500">{helper}</p></article>; }

function BarList({ rows, colors }: { rows: Distribution[]; colors: string[] }) { const max = Math.max(...rows.map((row) => row.count), 1); return <div className="space-y-4">{rows.map((row, index) => <div key={String(row.label)}><div className="mb-1.5 flex justify-between text-sm"><span className="font-medium text-slate-700">{row.label === 1 ? "Có toxic" : row.label === 0 ? "Không toxic" : row.label}</span><span className="text-slate-500">{formatNumber(row.count)} · {row.percentage.toFixed(1)}%</span></div><div className="h-2.5 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full ${colors[index % colors.length]}`} style={{ width: `${(row.count / max) * 100}%` }} /></div></div>)}</div>; }
function TopicBars({ rows }: { rows: TopicCount[] }) { const max = Math.max(...rows.map((row) => row.count), 1); return <div className="space-y-3">{rows.slice(0, 8).map((row) => <div key={row.topic} className="grid grid-cols-[130px_1fr_auto] items-center gap-3 text-sm"><span className="truncate font-medium text-slate-600">{row.topic}</span><div className="h-3 overflow-hidden rounded-full bg-rose-50"><div className="h-full rounded-full bg-rose-500" style={{ width: `${(row.count / max) * 100}%` }} /></div><span className="text-slate-500">{formatNumber(row.count)}</span></div>)}</div>; }
function GroupedBars({ rows, groupKey }: { rows: GroupCount[]; groupKey: "urgency" | "sentiment" }) { const topics = [...new Set(rows.map((row) => row.topic))].slice(0, 8); const groups = groupKey === "urgency" ? URGENCIES : SENTIMENTS; const colors = groupKey === "urgency" ? { high: "bg-rose-500", medium: "bg-amber-400", low: "bg-emerald-500" } : { negative: "bg-rose-500", neutral: "bg-slate-400", positive: "bg-emerald-500" }; return <div className="space-y-4">{topics.map((topic) => { const values = groups.map((group) => rows.find((row) => row.topic === topic && row[groupKey] === group)?.count || 0); const total = values.reduce((sum, value) => sum + value, 0) || 1; return <div key={topic}><div className="mb-1.5 text-sm font-medium text-slate-600">{topic}</div><div className="flex h-4 overflow-hidden rounded-full bg-slate-100">{values.map((value, index) => <div key={groups[index]} className={colors[groups[index] as keyof typeof colors]} style={{ width: `${(value / total) * 100}%` }} title={`${groups[index]}: ${value}`} />)}</div></div>; })}</div>; }
function Matrix({ rows }: { rows: GroupCount[] }) { const topics = [...new Set(rows.map((row) => row.topic))].slice(0, 8); return <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-left text-sm"><thead><tr className="border-b border-slate-200 text-slate-500"><th className="pb-3 font-medium">Topic</th>{SENTIMENTS.map((item) => <th className="pb-3 text-right font-medium" key={item}>{item}</th>)}</tr></thead><tbody>{topics.map((topic) => <tr className="border-b border-slate-100" key={topic}><td className="py-3 font-medium text-slate-700">{topic}</td>{SENTIMENTS.map((sentiment) => <td className="py-3 text-right" key={sentiment}>{formatNumber(rows.find((row) => row.topic === topic && row.sentiment === sentiment)?.count || 0)}</td>)}</tr>)}</tbody></table></div>; }
function Insights({ data }: { data: Analytics }) { const topNegative = data.negative_by_topic[0]; const high = count(data.urgency_distribution, "high"); const entries = [{ icon: AlertCircle, tone: "text-rose-600 bg-rose-50", title: topNegative ? `${topNegative.topic} có nhiều feedback negative nhất` : "Chưa có feedback negative", body: topNegative ? `${formatNumber(topNegative.count)} feedback cần xem xét.` : "" }, { icon: Activity, tone: "text-amber-600 bg-amber-50", title: `${formatNumber(high)} feedback high urgency`, body: "Ưu tiên xử lý các trường hợp cần phản hồi sớm." }, { icon: Wifi, tone: "text-blue-600 bg-blue-50", title: "Dùng Semantic Search để điều tra", body: "Tìm các phản hồi tương tự theo từng vấn đề cụ thể." }]; return <div className="space-y-3">{entries.map(({ icon: Icon, tone, title, body }) => <div className="flex gap-3 rounded-xl border border-slate-100 p-3" key={title}><div className={`h-9 w-9 shrink-0 rounded-lg p-2 ${tone}`}><Icon size={19} /></div><div><p className="text-sm font-semibold text-slate-700">{title}</p><p className="mt-1 text-xs leading-5 text-slate-500">{body}</p></div><ChevronRight size={17} className="ml-auto mt-2 text-slate-300" /></div>)}</div>; }
function PredictionResult({ result }: { result: Prediction }) { const rows = [{ label: "Sentiment", value: result.sentiment, score: result.sentiment_confidence }, { label: "Topic", value: result.topic, score: result.topic_confidence }, { label: "Urgency", value: result.urgency }, { label: "Toxic", value: result.toxic ? "Có" : "Không" }]; return <div className="grid gap-3 sm:grid-cols-2">{rows.map((row) => <div className="rounded-xl border border-slate-100 bg-slate-50 p-4" key={row.label}><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{row.label}</p><p className="mt-2 font-bold text-slate-800">{row.value}</p>{row.score !== undefined && <p className="mt-1 text-xs text-slate-500">Độ tin cậy {(row.score * 100).toFixed(1)}%</p>}</div>)}</div>; }
function ResultList({ results }: { results: SearchResult[] }) { if (!results.length) return <Empty text="Kết quả semantic search sẽ xuất hiện ở đây." />; return <div className="space-y-3">{results.map((row) => <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card" key={row.id}><div className="mb-3 flex flex-wrap gap-2"><Badge text={row.topic || "unknown"} tone="blue" /><Badge text={row.sentiment || "unknown"} tone="slate" /><Badge text={`rerank ${row.rerank_score.toFixed(3)}`} tone="teal" /></div><p className="text-sm leading-6 text-slate-700">{row.text}</p><p className="mt-3 text-xs text-slate-400">Vector score {row.vector_score.toFixed(3)} · Urgency {row.urgency || "unknown"}</p></article>)}</div>; }
