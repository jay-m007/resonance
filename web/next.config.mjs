/** @type {import('next').NextConfig} */
const base = process.env.PAGES_BASE_PATH || "";
export default {
  basePath: base,
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};
