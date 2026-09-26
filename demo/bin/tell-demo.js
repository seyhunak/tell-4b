#!/usr/bin/env node
// tell-demo: npx-runnable production server for the Tell-4B live demo.
// Usage:  cd demo && npm install && npm run build && npx .
//         TELL_PORT=4000 npx .
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const dir = path.join(__dirname, "..");
if (!fs.existsSync(path.join(dir, ".next"))) {
  console.error("tell-demo: no production build found.");
  console.error("Run `npm run build` first, then `npx .`.");
  process.exit(1);
}
const nextBin = path.join(dir, "node_modules", "next", "dist", "bin", "next");
if (!fs.existsSync(nextBin)) {
  console.error("tell-demo: dependencies missing. Run `npm install` first.");
  process.exit(1);
}
const port = process.env.TELL_PORT || "3000";
console.log(`tell-demo: live at http://localhost:${port}`);
const r = spawnSync(process.execPath, [nextBin, "start", "-p", port], {
  cwd: dir,
  stdio: "inherit",
});
process.exit(r.status ?? 0);
