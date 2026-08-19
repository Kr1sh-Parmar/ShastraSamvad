/** @type {import('next').NextConfig} */
export default {
  env: { NEXT_PUBLIC_API: process.env.NEXT_PUBLIC_API ?? "http://localhost:8000" },
};
