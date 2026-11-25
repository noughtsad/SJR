import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Job Search & Career Agent",
  description: "Hybrid semantic job search, resume-based recommendations and a career agent.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
