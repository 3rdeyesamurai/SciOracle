import { useState, useEffect } from 'react';
import { Blocks, Activity, Wallet, Cpu, Zap, Hash, Server, Hexagon } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import './index.css';

const API_BASE = "http://localhost:8000/api";
const SERVER_BASE = "http://localhost:8000";

function App() {
  const [walletInfo, setWalletInfo] = useState({ address: '...', balance: 0, symbol: 'POD', recent_transactions: [] });
  const [networkStats, setNetworkStats] = useState({ active_peers: 0, hash_rate: '0', dynamic_difficulty: 0, uptime: '0%' });
  const [blocks, setBlocks] = useState([]);
  const [systemStatus, setSystemStatus] = useState({ online: false, device: 'N/A' });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [wRes, nRes, bRes, sRes] = await Promise.all([
          fetch(`${API_BASE}/wallet`).catch(() => null),
          fetch(`${API_BASE}/network`).catch(() => null),
          fetch(`${API_BASE}/explorer/blocks?limit=20`).catch(() => null),
          fetch(`${API_BASE}/status`).catch(() => null)
        ]);

        if (wRes?.ok) setWalletInfo(await wRes.json());
        if (nRes?.ok) setNetworkStats(await nRes.json());
        if (bRes?.ok) {
          const bData = await bRes.json();
          setBlocks(bData.blocks || []);
        }
        if (sRes?.ok) {
          const sData = await sRes.json();
          setSystemStatus({ online: sData.status === 'online', device: sData.device });
        }
      } catch (err) {
        console.error("Dashboard sync error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Live real-time updates
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      {/* Sidebar / Wallet */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <div className="brand-icon"><Hexagon size={28} /></div>
          <h1>SciOracle PoD</h1>
        </div>

        <div className="status-card glass-panel" style={{ background: 'rgba(0,0,0,0.3)' }}>
          <div className="status-header">
            Crypto Wallet
            <Wallet size={16} color="var(--accent)" />
          </div>
          <div className="stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Local Address</span>
            <span className="code-text" style={{ fontSize: '0.85rem', wordBreak: 'break-all' }}>
              {walletInfo.address}
            </span>
          </div>
          <div className="stat-row">
            <span>Total Minted</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Zap size={18} color="#fbbf24" fill="#fbbf24" />
              <span className="stat-value" style={{ fontSize: '1.5rem', color: '#fbbf24' }}>
                {walletInfo.balance.toLocaleString()} {walletInfo.symbol}
              </span>
            </div>
          </div>
        </div>

        <div className="tx-history">
          <h3 className="section-title"><Activity size={16} /> Recent Transactions</h3>
          <div className="tx-list">
            {walletInfo.recent_transactions.map((tx, idx) => (
              <div key={idx} className="tx-item">
                <div className="tx-left">
                  <span className="tx-hash">tx_{tx.hash.substring(0, 8)}</span>
                  <span className="tx-reward">+{tx.reward} POD</span>
                </div>
                <div className="tx-right">
                  <span className="tx-time">Block #{tx.index}</span>
                </div>
              </div>
            ))}
            {walletInfo.recent_transactions.length === 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No recent transactions. Mine a block!</div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Node Dashboard Header */}
        <section className="dashboard-grid">
          <div className="dash-card glass-panel">
            <div className="dash-icon" style={{ background: 'rgba(99, 102, 241, 0.2)', color: '#818cf8' }}>
              <Server size={24} />
            </div>
            <div className="dash-info">
              <span className="dash-label">P2P Network Hash Rate</span>
              <span className="dash-value">{networkStats.hash_rate}</span>
            </div>
          </div>

          <div className="dash-card glass-panel">
            <div className="dash-icon" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}>
              <Activity size={24} />
            </div>
            <div className="dash-info">
              <span className="dash-label">Dynamic Difficulty (E Threshold)</span>
              <span className="dash-value">E &lt; {networkStats.dynamic_difficulty.toFixed(5)}</span>
            </div>
          </div>

          <div className="dash-card glass-panel">
            <div className="dash-icon" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>
              <Blocks size={24} />
            </div>
            <div className="dash-info">
              <span className="dash-label">Active WebSocket Peers</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="dash-value">{networkStats.active_peers} Nodes</span>
                <div className="pulsing-dot success"></div>
              </div>
            </div>
          </div>

          <div className="dash-card glass-panel">
            <div className="dash-icon" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24' }}>
              <Cpu size={24} />
            </div>
            <div className="dash-info">
              <span className="dash-label">Hardware VRAM Gate</span>
              <span className="dash-value" style={{ textTransform: 'uppercase' }}>{systemStatus.device}</span>
            </div>
          </div>
        </section>

        {/* Visual Block Explorer */}
        <section className="explorer-section">
          <h2 className="section-title" style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>
            <Hash size={24} color="var(--accent)" /> Visual Block Explorer
          </h2>

          <div className="blocks-container">
            {blocks.length === 0 ? (
              <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                Syncing blocks from local DiscoveryLedger...
              </div>
            ) : (
              blocks.map((block, idx) => (
                <div key={idx} className="block-card glass-panel">
                  <div className="block-header">
                    <div className="block-badge">Block #{block.height}</div>
                    <div className="block-meta">
                      <span className="hash-text" title={block.hash}>Hash: {block.hash.substring(0, 16)}...</span>
                      <span className="time-text">{new Date(block.timestamp * 1000).toLocaleString()}</span>
                    </div>
                  </div>

                  <div className="block-body">
                    <div className="math-pane">
                      <div className="math-label">Theorem Proven</div>
                      <div className="latex-container">
                        <BlockMath math={`${block.problem_math} = ${block.solution_math}`} />
                      </div>

                      <div className="energy-badge">
                        <Zap size={14} fill="currentColor" />
                        Energy: {block.energy.toFixed(5)}
                      </div>
                    </div>

                    {block.ast_chart ? (
                      <div className="ast-pane">
                        <div className="math-label" style={{ marginBottom: '0.5rem' }}>GCN Attention Mapping</div>
                        <img
                          src={`${SERVER_BASE}${block.ast_chart}`}
                          alt="AST Chart"
                          className="ast-image"
                        />
                      </div>
                    ) : (
                      <div className="ast-pane empty-ast">
                        <span>Genesis / No AST Chart</span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
