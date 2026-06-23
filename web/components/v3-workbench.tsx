"use client";

import { FormEvent, useEffect, useState } from "react";
import { Check, FileText, LoaderCircle, Sparkles, X } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const TOPICS = ["", "facilities", "teaching_learning", "student_services", "career_jobs", "events_news", "personal_social", "spam", "others"];

type ReviewItem = {
  feedback_id: string;
  text: string;
  dataset: string;
  topic: string;
  sentiment: string;
  urgency_predicted: string;
  urgency_final: string;
  reviewed: boolean;
  reviewer?: string;
  note?: string;
};

type Report = {
  id: number;
  title: string;
  content_markdown: string;
  created_at?: string;
};

type TopicCluster = {
  id: number;
  suggested_name: string;
  approved_name?: string;
  keywords: string[];
  examples: string[];
  size: number;
  status: "pending" | "approved" | "rejected";
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function ErrorMessage({ value }: { value: string | null }) {
  return value ? <p className="border border-rose-300 bg-rose-50 p-3 text-sm text-rose-700">{value}</p> : null;
}

function PageTitle({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="border-b border-[#d9d1c4] pb-5"><p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#315de0]">{eyebrow}</p><h2 className="mt-2 text-3xl font-bold tracking-tight text-[#142033]">{title}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">{description}</p></div>;
}

export function ReviewWorkbench() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [urgency, setUrgency] = useState("high");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const data = await request<{ items: ReviewItem[] }>(`/reviews?state=pending&urgency=${urgency}&limit=30`);
      setItems(data.items);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tải danh sách review."); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, [urgency]);

  const save = async (item: ReviewItem, urgencyFinal: string) => {
    setSaving(item.feedback_id); setError(null);
    try {
      await request(`/reviews/${encodeURIComponent(item.feedback_id)}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urgency_final: urgencyFinal, reviewer: "dashboard", note: "Đã duyệt trên dashboard" })
      });
      setItems((rows) => rows.filter((row) => row.feedback_id !== item.feedback_id));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể lưu review."); }
    finally { setSaving(null); }
  };

  return <div className="space-y-6"><PageTitle eyebrow="Human in the loop" title="Duyệt mức độ khẩn cấp" description="Sửa urgency mà không thay đổi CSV gốc. Giá trị được lưu trong SQLite và được analytics, report sử dụng ở các lần tải tiếp theo." />
    <section className="border border-[#d9d1c4] bg-[#fffefa] p-5"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-xl font-bold">Hàng đợi cần duyệt</h3><p className="mt-1 text-sm text-slate-500">Ưu tiên feedback có urgency cao trước.</p></div><select value={urgency} onChange={(event) => setUrgency(event.target.value)} className="border border-[#d9d1c4] bg-white px-3 py-2 text-sm"><option value="high">high</option><option value="medium">medium</option><option value="low">low</option></select></div></section>
    <ErrorMessage value={error} />
    {loading ? <div className="flex min-h-48 items-center justify-center border border-[#d9d1c4] bg-[#fffefa] text-sm text-slate-500"><LoaderCircle className="mr-2 animate-spin" size={18} />Đang tải feedback...</div> : !items.length ? <div className="border border-dashed border-[#d9d1c4] bg-[#faf8f2] p-10 text-center text-sm text-slate-500">Không còn feedback chưa duyệt với mức urgency này.</div> : <div className="space-y-4">{items.map((item) => <article key={item.feedback_id} className="border border-[#d9d1c4] bg-[#fffefa] p-5"><div className="flex flex-wrap items-center gap-2 text-xs font-semibold"><span className="bg-slate-100 px-2 py-1 text-slate-600">{item.topic}</span><span className="bg-rose-50 px-2 py-1 text-rose-700">{item.sentiment}</span><span className="bg-amber-50 px-2 py-1 text-amber-700">Dự đoán: {item.urgency_predicted}</span></div><p className="mt-4 text-sm leading-7 text-slate-700">{item.text}</p><div className="mt-5 flex flex-wrap items-center gap-2 border-t border-[#ece6dc] pt-4"><span className="mr-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Chốt urgency</span>{["high", "medium", "low"].map((value) => <button key={value} disabled={saving === item.feedback_id} onClick={() => save(item, value)} className={`border px-3 py-2 text-sm font-semibold ${value === "high" ? "border-rose-300 text-rose-700" : value === "medium" ? "border-amber-300 text-amber-700" : "border-emerald-300 text-emerald-700"}`}>{saving === item.feedback_id ? "Đang lưu..." : value}</button>)}</div></article>)}</div>}
  </div>;
}

export function ReportWorkbench() {
  const [topic, setTopic] = useState(""); const [title, setTitle] = useState("");
  const [reports, setReports] = useState<Report[]>([]); const [selected, setSelected] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null);
  const load = async () => { try { const data = await request<{ items: Report[] }>("/reports"); setReports(data.items); if (!selected && data.items[0]) setSelected(data.items[0]); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tải report."); } };
  useEffect(() => { load(); }, []);
  const generate = async (event: FormEvent) => { event.preventDefault(); setLoading(true); setError(null); try { const report = await request<Report>("/reports/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: title || undefined, topic: topic || undefined }) }); setReports((rows) => [report, ...rows]); setSelected(report); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tạo report."); } finally { setLoading(false); } };
  return <div className="space-y-6"><PageTitle eyebrow="Grounded reporting" title="Báo cáo tự động" description="Report được tổng hợp từ analytics và feedback đại diện đã rerank. Số liệu được tính bằng code; mô hình ngôn ngữ không tự tạo số liệu." />
    <form onSubmit={generate} className="grid gap-3 border border-[#d9d1c4] bg-[#fffefa] p-5 md:grid-cols-[1fr_220px_auto]"><input value={title} onChange={(event) => setTitle(event.target.value)} className="border border-[#d9d1c4] px-3 py-2.5 text-sm" placeholder="Tên report (không bắt buộc)" /><select value={topic} onChange={(event) => setTopic(event.target.value)} className="border border-[#d9d1c4] bg-white px-3 py-2.5 text-sm">{TOPICS.map((value) => <option key={value || "all"} value={value}>{value || "Tất cả chủ đề"}</option>)}</select><button disabled={loading} className="inline-flex items-center justify-center gap-2 bg-[#315de0] px-4 py-2.5 text-sm font-semibold text-white">{loading ? <LoaderCircle className="animate-spin" size={17} /> : <FileText size={17} />}{loading ? "Đang tạo" : "Tạo report"}</button></form>
    <ErrorMessage value={error} />
    <div className="grid gap-5 xl:grid-cols-[300px_1fr]"><aside className="border border-[#d9d1c4] bg-[#fffefa] p-3"><p className="px-2 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Report đã lưu</p>{reports.length ? reports.map((report) => <button key={report.id} onClick={() => setSelected(report)} className={`mb-1 w-full border-l-2 px-3 py-3 text-left text-sm ${selected?.id === report.id ? "border-[#315de0] bg-blue-50 text-[#142033]" : "border-transparent hover:bg-[#faf8f2]"}`}><span className="block font-semibold">{report.title}</span><span className="mt-1 block text-xs text-slate-500">{report.created_at || "Mới tạo"}</span></button>) : <p className="p-3 text-sm text-slate-500">Chưa có report nào.</p>}</aside><section className="min-h-96 border border-[#d9d1c4] bg-[#fffefa] p-6">{selected ? <pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-slate-700">{selected.content_markdown}</pre> : <p className="text-sm text-slate-500">Chọn hoặc tạo report để xem nội dung.</p>}</section></div>
  </div>;
}

export function TopicDiscoveryWorkbench() {
  const [topic, setTopic] = useState("others"); const [clusters, setClusters] = useState<TopicCluster[]>([]); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null);
  const load = async () => { try { const data = await request<{ items: TopicCluster[] }>("/topic-discovery/clusters?status=pending"); setClusters(data.items); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể tải cụm chủ đề."); } };
  useEffect(() => { load(); }, []);
  const run = async () => { setLoading(true); setError(null); try { const data = await request<{ clusters: TopicCluster[] }>("/topic-discovery/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic, max_items: 1200, min_cluster_size: 6 }) }); setClusters(data.clusters.filter((cluster) => cluster.status === "pending")); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể chạy topic discovery."); } finally { setLoading(false); } };
  const decide = async (cluster: TopicCluster, action: "approve" | "reject") => { setError(null); try { const path = action === "approve" ? `/topic-discovery/clusters/${cluster.id}/approve` : `/topic-discovery/clusters/${cluster.id}/reject`; await request(path, action === "approve" ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: cluster.suggested_name }) } : { method: "POST" }); setClusters((rows) => rows.filter((row) => row.id !== cluster.id)); } catch (reason) { setError(reason instanceof Error ? reason.message : "Không thể cập nhật cụm."); } };
  return <div className="space-y-6"><PageTitle eyebrow="Exploration" title="Khám phá chủ đề mới" description="Tạo cụm từ feedback chưa được phân loại rõ, xem các ví dụ đại diện, sau đó duyệt hoặc loại bỏ. Việc duyệt không tự ghi đè topic gốc." />
    <section className="flex flex-wrap items-center justify-between gap-4 border border-[#d9d1c4] bg-[#fffefa] p-5"><div><h3 className="text-xl font-bold">Chạy phân cụm</h3><p className="mt-1 text-sm text-slate-500">Dùng TF-IDF và MiniBatchKMeans trên tối đa 1.200 feedback.</p></div><div className="flex gap-3"><select value={topic} onChange={(event) => setTopic(event.target.value)} className="border border-[#d9d1c4] bg-white px-3 py-2 text-sm">{TOPICS.filter(Boolean).map((value) => <option key={value} value={value}>{value}</option>)}</select><button onClick={run} disabled={loading} className="inline-flex items-center gap-2 bg-[#315de0] px-4 py-2 text-sm font-semibold text-white">{loading ? <LoaderCircle className="animate-spin" size={17} /> : <Sparkles size={17} />}{loading ? "Đang chạy" : "Khám phá"}</button></div></section>
    <ErrorMessage value={error} />
    {!clusters.length ? <div className="border border-dashed border-[#d9d1c4] bg-[#faf8f2] p-10 text-center text-sm text-slate-500">Chưa có cụm đang chờ duyệt. Chạy discovery để tạo cụm mới.</div> : <div className="grid gap-4 xl:grid-cols-2">{clusters.map((cluster) => <article key={cluster.id} className="border border-[#d9d1c4] bg-[#fffefa] p-5"><div className="flex items-start justify-between gap-4"><div><p className="text-xs font-semibold uppercase tracking-wide text-[#315de0]">{cluster.size} feedback</p><h3 className="mt-1 text-xl font-bold">{cluster.suggested_name}</h3></div><span className="border border-amber-300 px-2 py-1 text-xs font-semibold text-amber-700">pending</span></div><div className="mt-4 flex flex-wrap gap-2">{cluster.keywords.map((keyword) => <span key={keyword} className="bg-slate-100 px-2 py-1 text-xs text-slate-600">{keyword}</span>)}</div><div className="mt-4 space-y-2 border-t border-[#ece6dc] pt-4">{cluster.examples.map((example) => <p key={example} className="border-l-2 border-[#315de0] pl-3 text-sm leading-6 text-slate-600">{example}</p>)}</div><div className="mt-5 flex gap-2"><button onClick={() => decide(cluster, "approve")} className="inline-flex items-center gap-1 border border-emerald-300 px-3 py-2 text-sm font-semibold text-emerald-700"><Check size={16} />Duyệt tên gợi ý</button><button onClick={() => decide(cluster, "reject")} className="inline-flex items-center gap-1 border border-rose-300 px-3 py-2 text-sm font-semibold text-rose-700"><X size={16} />Loại bỏ</button></div></article>)}</div>}
  </div>;
}
