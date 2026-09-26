import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tell-4B · Live Classification Demo",
  description: "50 preloaded tickets classified in real time by the Tell decision contract.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
