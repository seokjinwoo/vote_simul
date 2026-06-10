const $ = (id) => document.getElementById(id);

const inputs = {
  observedVotes: $("observedVotes"),
  otherVotes: $("otherVotes"),
  priorStrength: $("priorStrength"),
  districtA: $("districtA"),
  districtB: $("districtB"),
  tolerance: $("tolerance"),
  trials: $("trials"),
  seed: $("seed"),
};

const outputs = {
  priorStrengthValue: $("priorStrengthValue"),
  mainProbability: $("mainProbability"),
  mainOdds: $("mainOdds"),
  exactProbability: $("exactProbability"),
  exactCount: $("exactCount"),
  simProbability: $("simProbability"),
  simCount: $("simCount"),
  meanShare: $("meanShare"),
  posteriorInfo: $("posteriorInfo"),
  distributionCaption: $("distributionCaption"),
  collisionCaption: $("collisionCaption"),
  explanation: $("explanation"),
};

const charts = {
  distribution: $("distributionChart"),
  collision: $("collisionChart"),
  sweep: $("sweepChart"),
};

function readNumber(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function getParams() {
  const observed = Math.max(0, Math.round(readNumber(inputs.observedVotes, 3030)));
  const other = Math.max(0, Math.round(readNumber(inputs.otherVotes, 1440)));
  const prior = Math.max(0, readNumber(inputs.priorStrength, 0));
  const total = Math.max(1, observed + other);
  const share = observed / total;
  return {
    observed,
    other,
    prior,
    alpha: Math.max(0.000001, observed + prior * share),
    beta: Math.max(0.000001, other + prior * (1 - share)),
    districtA: clampInt(readNumber(inputs.districtA, 4470), 1, 50000),
    districtB: clampInt(readNumber(inputs.districtB, 4470), 1, 50000),
    tolerance: clampInt(readNumber(inputs.tolerance, 0), 0, 1000),
    trials: clampInt(readNumber(inputs.trials, 100000), 1000, 500000),
    seed: clampInt(readNumber(inputs.seed, 2026), 1, 2147483646),
  };
}

function clampInt(value, min, max) {
  return Math.min(max, Math.max(min, Math.round(value)));
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
  const base = logGamma(alpha + beta) - logGamma(alpha) - logGamma(beta);
  const tail = logGamma(n + alpha + beta);
  let maxLog = -Infinity;
  const logs = new Float64Array(n + 1);
  for (let k = 0; k <= n; k += 1) {
    const value =
      logChoose(n, k) +
      logGamma(k + alpha) +
      logGamma(n - k + beta) -
      tail +
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

function collisionProbability(pmfA, pmfB, tolerance) {
  let probability = 0;
  const contributions = [];
  for (let a = 0; a < pmfA.length; a += 1) {
    const start = Math.max(0, a - tolerance);
    const end = Math.min(pmfB.length - 1, a + tolerance);
    let window = 0;
    for (let b = start; b <= end; b += 1) window += pmfB[b];
    const contribution = pmfA[a] * window;
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

function simulate(params) {
  const rng = makeRng(params.seed);
  let matches = 0;
  for (let i = 0; i < params.trials; i += 1) {
    const pA = betaSample(params.alpha, params.beta, rng);
    const pB = betaSample(params.alpha, params.beta, rng);
    const a = binomialSample(params.districtA, pA, rng);
    const b = binomialSample(params.districtB, pB, rng);
    if (Math.abs(a - b) <= params.tolerance) matches += 1;
  }
  return { matches, probability: matches / params.trials };
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
  const width = Math.max(320, Math.round(rect.width * ratio));
  const height = Math.round((Number(canvas.getAttribute("height")) || 360) * (rect.width / (Number(canvas.getAttribute("width")) || 900)) * ratio);
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
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 56, right: 22, top: 24, bottom: 42 };
  drawAxes(ctx, width, height, pad);
  const all = series.flatMap((s) => s.points);
  const minX = options.minX ?? Math.min(...all.map((p) => p.x));
  const maxX = options.maxX ?? Math.max(...all.map((p) => p.x));
  const maxY = options.maxY ?? Math.max(...all.map((p) => p.y), 0.000001);
  const xScale = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * (width - pad.left - pad.right);
  const yScale = (y) => height - pad.bottom - (y / maxY) * (height - pad.top - pad.bottom);

  ctx.fillStyle = "#667077";
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillText(options.yLabel || "확률", 8, pad.top + 4);
  ctx.fillText(String(Math.round(minX)), pad.left, height - 12);
  ctx.fillText(String(Math.round(maxX)), width - pad.right - 48, height - 12);

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2.5 * (window.devicePixelRatio || 1);
    ctx.beginPath();
    item.points.forEach((point, index) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function drawBars(canvas, points, color, options = {}) {
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 56, right: 22, top: 24, bottom: 42 };
  drawAxes(ctx, width, height, pad);
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
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillText(options.yLabel || "확률", 8, pad.top + 4);
  ctx.fillText(String(minX), pad.left, height - 12);
  ctx.fillText(String(maxX), width - pad.right - 48, height - 12);
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

function sweep(params) {
  const center = Math.round((params.districtA + params.districtB) / 2);
  const low = Math.max(50, Math.round(center * 0.35));
  const high = Math.min(50000, Math.round(center * 1.65));
  const step = Math.max(25, Math.round((high - low) / 34));
  const points = [];
  for (let n = low; n <= high; n += step) {
    const pmf = betaBinomialPmf(n, params.alpha, params.beta);
    points.push({ x: n, y: collisionProbability(pmf, pmf, params.tolerance).probability });
  }
  return points;
}

function render() {
  const params = getParams();
  inputs.observedVotes.value = params.observed;
  inputs.otherVotes.value = params.other;
  inputs.districtA.value = params.districtA;
  inputs.districtB.value = params.districtB;
  inputs.tolerance.value = params.tolerance;
  inputs.trials.value = params.trials;
  inputs.seed.value = params.seed;
  outputs.priorStrengthValue.value = params.prior;

  const pmfA = betaBinomialPmf(params.districtA, params.alpha, params.beta);
  const pmfB = betaBinomialPmf(params.districtB, params.alpha, params.beta);
  const exact = collisionProbability(pmfA, pmfB, params.tolerance);
  const sim = simulate(params);
  const mean = params.alpha / (params.alpha + params.beta);

  outputs.mainProbability.textContent = percent(exact.probability);
  outputs.mainOdds.textContent = odds(exact.probability);
  outputs.exactProbability.textContent = percent(exact.probability);
  outputs.exactCount.textContent = `${odds(exact.probability)} 수준`;
  outputs.simProbability.textContent = percent(sim.probability);
  outputs.simCount.textContent = `${number(params.trials)}회 중 ${number(sim.matches)}회 일치`;
  outputs.meanShare.textContent = percent(mean);
  outputs.posteriorInfo.textContent = `Beta(${params.alpha.toFixed(1)}, ${params.beta.toFixed(1)})`;
  outputs.distributionCaption.textContent = `A ${number(params.districtA)}표, B ${number(params.districtB)}표`;
  outputs.collisionCaption.textContent = params.tolerance === 0 ? "정확히 같은 득표수" : `±${params.tolerance}표 이내`;
  outputs.explanation.textContent =
    `현재 설정에서는 두 지역구의 득표수가 ${params.tolerance === 0 ? "정확히 같을" : `±${params.tolerance}표 이내일`} ` +
    `확률이 ${percent(exact.probability)}입니다. 이는 ${odds(exact.probability)} 정도이며, ` +
    `몬테카를로 ${number(params.trials)}회 반복 결과는 ${percent(sim.probability)}로 나왔습니다.`;

  drawLineChart(charts.distribution, [
    { color: "#0f766e", points: downsamplePmf(pmfA).map((p) => ({ x: p.x, y: p.y })) },
    { color: "#b45309", points: downsamplePmf(pmfB).map((p) => ({ x: p.x, y: p.y })) },
  ]);
  drawBars(charts.collision, topWindow(exact.contributions), "#375a9e", { yLabel: "기여" });
  drawLineChart(charts.sweep, [{ color: "#9f1239", points: sweep(params) }], { yLabel: "일치 확률" });
}

let renderTimer = 0;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(render, 80);
}

for (const input of Object.values(inputs)) {
  input.addEventListener("input", scheduleRender);
}
$("runButton").addEventListener("click", render);
window.addEventListener("resize", scheduleRender);

render();
