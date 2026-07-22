import * as esbuild from 'esbuild';

console.log('Starting demo development server and bundler...');

const ctx = await esbuild.context({
  entryPoints: [
    { in: 'src/app.ts', out: 'app' },
    { in: 'src/worker.ts', out: 'worker' }
  ],
  bundle: true,
  minify: true,
  outdir: 'static',
});

await ctx.watch();
const { port } = await ctx.serve({
  servedir: 'static',
  port: 3000,
});

console.log(`Demo server running on http://localhost:${port}`);
