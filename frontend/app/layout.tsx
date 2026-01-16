import "./globals.css";

export const metadata = {
  title: "Clinical Guideline Chat",
  description: "Clinician-facing guideline assistant with streaming answers"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="et">
      <body>{children}</body>
    </html>
  );
}
