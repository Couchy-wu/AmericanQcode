/**
 * Stock Screener page interactivity.
 */

const SCREENER_BODY = document.getElementById('screener-body');
const BTN_SCAN = document.getElementById('btn-scan');
const FILTER_STRATEGY = document.getElementById('filter-strategy');
const FILTER_DIRECTION = document.getElementById('filter-direction');
const FILTER_MIN_RSI = document.getElementById('filter-min-rsi');
const FILTER_MAX_RSI = document.getElementById('filter-max-rsi');

async function runScreener() {
    if (!SCREENER_BODY) return;

    SCREENER_BODY.innerHTML = '<tr><td colspan="8" class="empty-state">Scanning...</td></tr>';

    const params = new URLSearchParams({ watchlist: 'sp500_sample' });

    if (FILTER_STRATEGY.value) params.set('strategy', FILTER_STRATEGY.value);
    if (FILTER_DIRECTION.value) params.set('direction', FILTER_DIRECTION.value);
    if (FILTER_MIN_RSI.value) params.set('min_rsi', FILTER_MIN_RSI.value);
    if (FILTER_MAX_RSI.value) params.set('max_rsi', FILTER_MAX_RSI.value);

    try {
        const resp = await fetch(`/api/screener?${params.toString()}`);
        if (!resp.ok) throw new Error('Screener API error');
        const data = await resp.json();

        if (!data.results || data.results.length === 0) {
            SCREENER_BODY.innerHTML = '<tr><td colspan="8" class="empty-state">No results match your filters</td></tr>';
            return;
        }

        SCREENER_BODY.innerHTML = data.results.map(row => {
            const changeClass = row.change_pct >= 0 ? 'color: #00c853' : 'color: #ff1744';
            const macdClass = row.macd_signal === 'Buy' ? 'color: #00c853' :
                             (row.macd_signal === 'Sell' ? 'color: #ff1744' : '');
            const maClass = row.ma_cross === 'Golden' ? 'color: #00c853' :
                           (row.ma_cross === 'Death' ? 'color: #ff1744' : '');
            return `
                <tr>
                    <td><a href="/chart/${row.ticker}" style="color:#448aff;font-weight:600;">${row.ticker}</a></td>
                    <td>$${row.price.toFixed(2)}</td>
                    <td style="${changeClass}">${row.change_pct >= 0 ? '+' : ''}${row.change_pct.toFixed(2)}%</td>
                    <td>${row.rsi}</td>
                    <td style="${macdClass}">${row.macd_signal}</td>
                    <td>${(row.volume / 1000000).toFixed(1)}M</td>
                    <td>${row.adx}</td>
                    <td style="${maClass}">${row.ma_cross}</td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        SCREENER_BODY.innerHTML = `<tr><td colspan="8" class="empty-state">Error: ${err.message}</td></tr>`;
    }
}

if (BTN_SCAN) BTN_SCAN.addEventListener('click', runScreener);

// Load signals page
const SIGNALS_BODY = document.getElementById('signals-body');
const BTN_REFRESH = document.getElementById('btn-refresh');

if (SIGNALS_BODY || BTN_REFRESH) {
    async function loadSignals() {
        if (!SIGNALS_BODY) return;
        const params = new URLSearchParams({ limit: '50' });
        if (FILTER_STRATEGY && FILTER_STRATEGY.value) params.set('strategy', FILTER_STRATEGY.value);
        if (FILTER_DIRECTION && FILTER_DIRECTION.value) params.set('direction', FILTER_DIRECTION.value);

        try {
            const resp = await fetch(`/api/signals?${params.toString()}`);
            if (!resp.ok) throw new Error('Signals API error');
            const signals = await resp.json();

            if (!signals || signals.length === 0) {
                SIGNALS_BODY.innerHTML = '<tr><td colspan="7" class="empty-state">No signals found</td></tr>';
                return;
            }

            SIGNALS_BODY.innerHTML = signals.map(s => {
                const isBull = s.direction === 'BULLISH';
                return `
                    <tr>
                        <td>${s.timestamp ? new Date(s.timestamp).toLocaleString() : '-'}</td>
                        <td><a href="/chart/${s.ticker}" style="color:#448aff;">${s.ticker}</a></td>
                        <td style="color:${isBull ? '#00c853' : '#ff1744'};font-weight:600;">
                            ${isBull ? '▲ BULL' : '▼ BEAR'}
                        </td>
                        <td>${(s.confidence * 100).toFixed(0)}%</td>
                        <td>${s.strategy}</td>
                        <td>$${s.price?.toFixed(2) || '-'}</td>
                        <td style="color:#a0a0b0;">${s.reasoning || ''}</td>
                    </tr>
                `;
            }).join('');
        } catch (err) {
            SIGNALS_BODY.innerHTML = `<tr><td colspan="7" class="empty-state">Error: ${err.message}</td></tr>`;
        }
    }

    if (BTN_REFRESH) BTN_REFRESH.addEventListener('click', loadSignals);
    loadSignals();
}
