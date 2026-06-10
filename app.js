const $ = (id) => document.getElementById(id);

const inputs = {
  observedVotes: $("observedVotes"),
  otherVotes: $("otherVotes"),
  priorStrength: $("priorStrength"),
  tolerance: $("tolerance"),
  trials: $("trials"),
  seed: $("seed"),
};

const outputs = {
  priorStrengthValue: $("priorStrengthValue"),
  districtTotal: $("districtTotal"),
  mainProbability: $("mainProbability"),
  mainOdds: $("mainOdds"),
  exactProbability: $("exactProbability"),
  exactCount: $("exactCount"),
  simProbability: $("simProbability"),
  simCount: $("simCount"),
  meanShare: $("meanShare"),
  posteriorInfo: $("posteriorInfo"),
  progressCaption: $("progressCaption"),
  progressBar: $("progressBar"),
  distributionCaption: $("distributionCaption"),
  collisionCaption: $("collisionCaption"),
  explanation: $("explanation"),
};

const charts = {
  animation: $("animationChart"),
  distribution: $("distributionChart"),
  collision: $("collisionChart"),
};

let activeRun = 0;
let lastStaticRender = null;

function readNumber(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function clampInt(value, min, max) {
  return Math.min(max, Math.max(min, Math.round(value)));
}

function getParams() {
  const observed = Math.max(0, Math.round(readNumber(inputs.observedVotes, 3030)));
  const other = Math.max(0, Math.round(readNumber(inputs.otherVotes, 1440)));
  const totalVotes = Math.max(1, observed + other);
  const prior = Math.max(0, readNumber(inputs.priorStrength, 0));
  const share = observed / totalVotes;

  return {
    observed,
    other,
    totalVotes,
    prior,
    alpha: Math.max(0.000001, observed + prior * share),
    beta: Math.max(0.000001, other + prior * (1 - share)),
    tolerance: clampInt(readNumber(inputs.tolerance, 0), 0, 1000),
    trials: clampInt(readNumber(inputs.trials, 100000), 1000, 500000),
    seed: clampInt(readNumber(inputs.seed, 2026), 1, 2147483646),
  };
}

function logGamma(z) {
  const p = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = 0.9999999999998099;
  for (let i = 0; i < p.length; i += 1) x += p[i] / (z + i + 1);
  const t = z + p.length - 0.5;
  return 0.9189385332046727 + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function logChoose(n, k) {
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
}

function betaBinomialPmf(n, alpha, beta) {
  const result = new Float64Array(n + 1);
  const logs = new Float64Array(n + 1);
  const base = logGamma(alpha + beta) - logGamma(alpha) - logGamma(beta);
  const denominator = logGamma(n + alpha + beta);
  let maxLog = -Infinity;

  for (let k = 0; k <= n; k += 1) {
    const value =
      logChoose(n, k) +
      logGamma(k + alpha) +
      logGamma(n - k + beta) -
      denominator +
      base;
    logs[k] = value;
    if (value > maxLog) maxLog = value;
  }

  let sum = 0;
  for (let k = 0; k <= n; k += 1) {
    result[k] = Math.exp(logs[k] - maxLog);
    sum += result[k];
  }
  for (let k = 0; k <= n; k += 1) result[k] /= sum;
  return result;
}

function collisionProbability(pmf, tolerance) {
  let probability = 0;
  const contributions = [];
  for (let a = 0; a < pmf.length; a += 1) {
    const start = Math.max(0, a - tolerance);
    const end = Math.min(pmf.length - 1, a + tolerance);
    let window = 0;
    for (let b = start; b <= end; b += 1) window += pmf[b];
    const contribution = pmf[a] * window;
    probability += contribution;
    if (contribution > 0) contributions.push({ k: a, p: contribution });
  }
  return { probability, contributions };
}

function makeRng(seed) {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function normalSample(rng) {
  const u1 = Math.max(Number.MIN_VALUE, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gammaSample(shape, rng) {
  if (shape < 1) {
    return gammaSample(shape + 1, rng) * Math.pow(Math.max(Number.MIN_VALUE, rng()), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = normalSample(rng);
    let v = 1 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function betaSample(alpha, beta, rng) {
  const x = gammaSample(alpha, rng);
  const y = gammaSample(beta, rng);
  return x / (x + y);
}

function binomialSample(n, p, rng) {
  if (n <= 80) {
    let count = 0;
    for (let i = 0; i < n; i += 1) if (rng() < p) count += 1;
    return count;
  }
  const mean = n * p;
  const sd = Math.sqrt(Math.max(0.000001, n * p * (1 - p)));
  return clampInt(Math.round(mean + sd * normalSample(rng)), 0, n);
}

function percent(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 0.01) return `${(value * 100).toFixed(2)}%`;
  if (value >= 0.001) return `${(value * 100).toFixed(3)}%`;
  return `${(value * 100).toPrecision(3)}%`;
}

function number(value) {
  return new Intl.NumberFormat("ko-KR").format(Math.round(value));
}

function odds(value) {
  if (!value) return "거의 0";
  return `약 ${number(1 / value)}번 중 1번`;
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const baseWidth = Number(canvas.getAttribute("width")) || 900;
  const baseHeight = Number(canvas.getAttribute("height")) || 360;
  const width = Math.max(320, Math.round(rect.width * ratio));
  const height = Math.round(baseHeight * (rect.width / baseWidth) * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { ctx: canvas.getContext("2d"), width, height, ratio };
}

function drawAxes(ctx, width, height, padding) {
  ctx.strokeStyle = "#d7ddd8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
}

function drawLineChart(canvas, series, options = {}) {
  const { ctx, width, height, ratio } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = {
    left: 56 * ratio,
    right: 22 * ratio,
    top: 24 * ratio,
    bottom: 42 * ratio,
  };
  drawAxes(ctx, width, height, pad);

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return;
  const minX = options.minX ?? Math.min(...all.map((p) => p.x));
  const maxX = options.maxX ?? Math.max(...all.map((p) => p.x));
  const maxY = options.maxY ?? Math.max(...all.map((p) => p.y), 0.000001);
  const xScale = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - (y / maxY) * (height - pad.top - pad.bottom);

  ctx.fillStyle = "#667077";
  ctx.font = `${12 * ratio}px Arial`;
  ctx.fillText(options.yLabel || "확률", 8 * ratio, pad.top + 4 * ratio);
  ctx.fillText(String(Math.round(minX)), pad.left, height - 12 * ratio);
  ctx.fillText(String(Math.round(maxX)), width - pad.right - 54 * ratio, height - 12 * ratio);

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = (item.width || 2.5) * ratio;
    ctx.setLineDash(item.dash ? item.dash.map((v) => v * ratio) : []);
    ctx.beginPath();
    item.points.forEach((point, index) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawBars(canvas, points, color, options = {}) {
  const { ctx, width, height, ratio } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = {
    left: 56 * ratio,
    right: 22 * ratio,
    top: 24 * ratio,
    bottom: 42 * ratio,
  };
  drawAxes(ctx, width, height, pad);
  if (points.length === 0) return;

  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const maxY = Math.max(...points.map((p) => p.y), 0.000001);
  const plotW = width - pad.left - pad.right;
  const xScale = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * plotW;
  const yScale = (y) => height - pad.bottom - (y / maxY) * (height - pad.top - pad.bottom);
  const barW = Math.max(1, plotW / Math.max(1, points.length) - 1);

  ctx.fillStyle = color;
  for (const point of points) {
    const x = xScale(point.x);
    const y = yScale(point.y);
    ctx.fillRect(x, y, barW, height - pad.bottom - y);
  }

  ctx.fillStyle = "#667077";
  ctx.font = `${12 * ratio}px Arial`;
  ctx.fillText(options.yLabel || "확률", 8 * ratio, pad.top + 4 * ratio);
  ctx.fillText(String(minX), pad.left, height - 12 * ratio);
  ctx.fillText(String(maxX), width - pad.right - 54 * ratio, height - 12 * ratio);
}

function downsamplePmf(pmf, limit = 220) {
  const points = [];
  const step = Math.max(1, Math.ceil(pmf.length / limit));
  for (let i = 0; i < pmf.length; i += step) {
    let y = 0;
    const end = Math.min(pmf.length, i + step);
    for (let k = i; k < end; k += 1) y += pmf[k];
    points.push({ x: i + Math.floor((end - i) / 2), y });
  }
  return points;
}

function topWindow(points, limit = 180) {
  const sorted = [...points].sort((a, b) => b.p - a.p).slice(0, limit);
  const min = Math.min(...sorted.map((p) => p.k));
  const max = Math.max(...sorted.map((p) => p.k));
  return points
    .filter((p) => p.k >= min && p.k <= max)
    .map((p) => ({ x: p.k, y: p.p }));
}

function renderStatic(params) {
  const pmf = betaBinomialPmf(params.totalVotes, params.alpha, params.beta);
  const exact = collisionProbability(pmf, params.tolerance);
  const mean = params.alpha / (params.alpha + params.beta);

  inputs.observedVotes.value = params.observed;
  inputs.otherVotes.value = params.other;
  inputs.tolerance.value = params.tolerance;
  inputs.trials.value = params.trials;
  inputs.seed.value = params.seed;
  outputs.priorStrengthValue.textContent = params.prior;
  outputs.districtTotal.textContent = number(params.totalVotes);

  outputs.mainProbability.textContent = percent(exact.probability);
  outputs.mainOdds.textContent = odds(exact.probability);
  outputs.exactProbability.textContent = percent(exact.probability);
  outputs.exactCount.textContent = `${odds(exact.probability)} 수준`;
  outputs.meanShare.textContent = percent(mean);
  outputs.posteriorInfo.textContent = `Beta(${params.alpha.toFixed(1)}, ${params.beta.toFixed(1)})`;
  outputs.distributionCaption.textContent = `각 지역구 ${number(params.totalVotes)}표`;
  outputs.collisionCaption.textContent = params.tolerance === 0 ? "정확히 같은 득표수" : `±${params.tolerance}표 이내`;
  outputs.explanation.textContent =
    `후보 ${number(params.observed)}표와 상대 ${number(params.other)}표의 합계 ${number(params.totalVotes)}표를 ` +
    `각 지역구 총투표수로 적용했습니다. 확률은 ${percent(exact.probability)}이며, ` +
    `${odds(exact.probability)} 정도입니다.`;

  drawLineChart(charts.distribution, [{ color: "#0f766e", points: downsamplePmf(pmf) }]);
  drawBars(charts.collision, topWindow(exact.contributions), "#375a9e", { yLabel: "기여" });

  lastStaticRender = { params, exact };
  return exact;
}

function drawSimulation(points, exactProbability, trials) {
  const maxObserved = Math.max(exactProbability, ...points.map((p) => p.y), 0.000001);
  drawLineChart(
    charts.animation,
    [
      { color: "#0f766e", width: 3, points },
      {
        color: "#b45309",
        width: 2,
        dash: [7, 5],
        points: [
          { x: 0, y: exactProbability },
          { x: trials, y: exactProbability },
        ],
      },
    ],
    { minX: 0, maxX: trials, maxY: maxObserved * 1.2, yLabel: "누적 비율" }
  );
}

function runAnimatedSimulation(params, exactProbability) {
  const runId = ++activeRun;
  const rng = makeRng(params.seed);
  const points = [{ x: 0, y: 0 }];
  const batch = Math.max(250, Math.min(5000, Math.round(params.trials / 80)));
  let done = 0;
  let matches = 0;

  outputs.progressBar.style.width = "0%";
  outputs.simProbability.textContent = "0.000%";
  outputs.simCount.textContent = `0회 중 0회 일치`;
  outputs.progressCaption.textContent = `0 / ${number(params.trials)}회`;
  drawSimulation(points, exactProbability, params.trials);

  function step() {
    if (runId !== activeRun) return;
    const end = Math.min(params.trials, done + batch);
    for (; done < end; done += 1) {
      const pA = betaSample(params.alpha, params.beta, rng);
      const pB = betaSample(params.alpha, params.beta, rng);
      const a = binomialSample(params.totalVotes, pA, rng);
      const b = binomialSample(params.totalVotes, pB, rng);
      if (Math.abs(a - b) <= params.tolerance) matches += 1;
    }

    const probability = matches / done;
    points.push({ x: done, y: probability });
    outputs.simProbability.textContent = percent(probability);
    outputs.simCount.textContent = `${number(done)}회 중 ${number(matches)}회 일치`;
    outputs.progressCaption.textContent = `${number(done)} / ${number(params.trials)}회`;
    outputs.progressBar.style.width = `${(done / params.trials) * 100}%`;
    drawSimulation(points, exactProbability, params.trials);

    if (done < params.trials) {
      requestAnimationFrame(step);
    } else {
      outputs.progressCaption.textContent = `완료: ${number(params.trials)}회`;
    }
  }

  requestAnimationFrame(step);
}

function renderAndRun() {
  const params = getParams();
  const exact = renderStatic(params);
  runAnimatedSimulation(params, exact.probability);
}

let renderTimer = 0;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(renderAndRun, 120);
}

for (const input of Object.values(inputs)) {
  input.addEventListener("input", scheduleRender);
}

$("runButton").addEventListener("click", renderAndRun);
window.addEventListener("resize", () => {
  if (!lastStaticRender) return;
  renderStatic(lastStaticRender.params);
});

renderAndRun();
