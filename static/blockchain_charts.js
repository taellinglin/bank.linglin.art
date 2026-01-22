// blockchain_charts.js
// Draws ring charts and a mini timeline for Blockchain Viewer Activity tab

// --- RING CHARTS ---
function drawRingChart(canvas, value, max, color, label, sublabel) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const cx = w/2, cy = h/2, r = Math.min(w, h)/2 - 10;
    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2*Math.PI);
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 16;
    ctx.stroke();
    // Value arc
    const frac = Math.max(0, Math.min(1, value/max));
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI/2, -Math.PI/2 + frac*2*Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.stroke();
    // Text
    ctx.font = 'bold 1.2rem sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(value, cx, cy-6);
    ctx.font = '0.9rem sans-serif';
    ctx.fillStyle = '#aaa';
    ctx.fillText(label, cx, cy+14);
    if (sublabel) ctx.fillText(sublabel, cx, cy+32);
}

// --- MINI TIMELINE ---
function drawMiniTimeline(canvas, events, color) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    // Draw vertical lines for events
    const n = events.length;
    if (n === 0) return;
    let minX = w, maxX = 0;
    for (let i=0; i<n; ++i) {
        const x = Math.round(i/(n-1) * (w-40)) + 20;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        ctx.beginPath();
        ctx.moveTo(x, 20);
        ctx.lineTo(x, h-20);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }
    // Timeline base
    ctx.beginPath();
    ctx.moveTo(minX, h/2);
    ctx.lineTo(maxX, h/2);
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 2;
    ctx.stroke();
}

window.renderBlockchainRingsAndTimeline = function(stats, timelineEvents) {
    // Ring charts
    drawRingChart(document.getElementById('ringBlocks'), stats.blocks, stats.blocksMax, '#32d7ee', 'Blocks');
    drawRingChart(document.getElementById('ringTxs'), stats.txs, stats.txsMax, '#bf5af2', 'TXs');
    drawRingChart(document.getElementById('ringGenesis'), stats.genesis, stats.genesisMax, '#30d158', 'Genesis');
    drawRingChart(document.getElementById('ringRewards'), stats.rewards, stats.rewardsMax, '#ffd60a', 'Rewards');
    // Mini timeline
    drawMiniTimeline(document.getElementById('miniTimeline'), timelineEvents, '#5ac8fa');
};


// --- CLEAN TIMELINE ---
// timelineEvents: [{type: 'transfer'|'genesis'|'reward', block: n}, ...]
window.renderCleanTimeline = function(canvas, timelineEvents) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    // --- GitHub-style activity grid ---
    // 1. Build a day-indexed map for each type
    const colorMap = {reward:'#ffd60a', genesis:'#30d158', transfer:'#bf5af2'};
    const types = ['reward','genesis','transfer'];
    // Find min/max block and min/max date
    let minDate = null, maxDate = null;
    const dayMap = {reward:{}, genesis:{}, transfer:{}};
    for (const ev of timelineEvents) {
        if (!ev.date) continue;
        const d = ev.date.split('T')[0];
        if (!minDate || d < minDate) minDate = d;
        if (!maxDate || d > maxDate) maxDate = d;
        if (!dayMap[ev.type][d]) dayMap[ev.type][d] = 0;
        dayMap[ev.type][d]++;
    }
    if (!minDate || !maxDate) {
        // fallback: draw empty grid
        ctx.save();
        ctx.font = 'bold 1rem sans-serif';
        ctx.fillStyle = '#aaa';
        ctx.textAlign = 'center';
        ctx.fillText('No activity events', w/2, h/2 + 8);
        ctx.restore();
        return;
    }
    // Build date array (weeks x days)
    const start = new Date(minDate);
    const end = new Date(maxDate);
    const days = [];
    for (let d = new Date(start); d <= end; d.setDate(d.getDate()+1)) {
        days.push(new Date(d));
    }
    // Layout: 3 rows (reward/genesis/transfer), columns = days
    const cell = Math.min(14, Math.max(8, Math.floor((w-80)/days.length)));
    const gridW = cell * days.length;
    const gridH = cell * 3;
    const offsetX = Math.max(40, (w-gridW)/2);
    const offsetY = Math.max(32, (h-gridH)/2);
    // Draw grid
    for (let row=0; row<3; ++row) {
        const type = types[row];
        for (let col=0; col<days.length; ++col) {
            const dstr = days[col].toISOString().split('T')[0];
            const count = dayMap[type][dstr] || 0;
            ctx.save();
            ctx.beginPath();
            ctx.rect(offsetX+col*cell, offsetY+row*cell, cell-2, cell-2);
            ctx.fillStyle = count ? colorMap[type] : '#222';
            ctx.globalAlpha = count ? Math.min(0.3+0.15*count, 1) : 0.18;
            ctx.fill();
            ctx.restore();
        }
    }
    // Row labels
    ctx.save();
    ctx.font = '0.95rem sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = colorMap.reward;
    ctx.fillText('Mining Reward', offsetX-8, offsetY+cell*0.7);
    ctx.fillStyle = colorMap.genesis;
    ctx.fillText('Genesis', offsetX-8, offsetY+cell*1.7);
    ctx.fillStyle = colorMap.transfer;
    ctx.fillText('Transfer', offsetX-8, offsetY+cell*2.7);
    ctx.restore();
    // Date labels
    ctx.save();
    ctx.font = '0.85rem sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#aaa';
    for (let col=0; col<days.length; col+=Math.ceil(days.length/8)) {
        const d = days[col];
        const label = d.toISOString().split('T')[0];
        ctx.fillText(label, offsetX+col*cell+cell/2, offsetY-6);
    }
    ctx.restore();
};

// Example usage (replace with real data from backend):
// window.renderBlockchainRingsAndTimeline({blocks: 100, blocksMax: 200, txs: 150, txsMax: 300, genesis: 20, genesisMax: 50, rewards: 10, rewardsMax: 20}, [1,2,3,4,5]);
