import "./globals.css";

export const metadata = {
  title: "Shastra Samvad",
  description: "Offline scripture-grounded Guru",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
