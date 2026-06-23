import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Student Voice Intelligence",
  description: "Phân tích phản hồi sinh viên"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
