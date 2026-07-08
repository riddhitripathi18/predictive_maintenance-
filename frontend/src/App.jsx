import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, Clock, Database, FileText, Settings2, Shield, 
  AlertTriangle, CheckCircle2, RefreshCw, Plus, Trash2, 
  Download, Upload, AlertCircle, Thermometer, ShieldAlert, Cpu
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import Gauge from './components/Gauge';

const API_BASE = 'http://localhost:8000/api';

const TYPE_MAP = {
  "L (Low)": "L",
  "M (Medium)": "M",
  "H (High)": "H"
};

export default function App() {
  // Navigation & Config States
  const [activeTab, setActiveTab] = useState('simulator');
  const [opMode, setOpMode] = useState('Balanced');
  const [config, setConfig] = useState(null);
  const [activeThresh, setActiveThresh] = useState(0.35);

  // ── TAB 1: SIMULATOR STATES ─────────────────────────────────
  const [simType, setSimType] = useState('L (Low)');
  const [simAir, setSimAir] = useState(300.0);
  const [simProc, setSimProc] = useState(310.0);
  const [simRpm, setSimRpm] = useState(1500);
  const [simTorque, setSimTorque] = useState(40.0);
  const [simWear, setSimWear] = useState(108);
  const [simResult, setSimResult] = useState(null);

  // ── TAB 2: FORECAST STATES ──────────────────────────────────
  const [foreType, setForeType] = useState('L (Low)');
  const [foreAir, setForeAir] = useState(300.0);
  const [foreProc, setForeProc] = useState(310.0);
  const [foreRpm, setForeRpm] = useState(1500);
  const [foreTorque, setForeTorque] = useState(38.0);
  const [foreWear, setForeWear] = useState(80);
  const [foreHorizon, setForeHorizon] = useState(12);
  const [foreWearRate, setForeWearRate] = useState(12.0);
  const [foreRpmDrift, setForeRpmDrift] = useState(-25.0);
  const [foreTorqDrift, setForeTorqDrift] = useState(1.2);
  const [scRpm, setScRpm] = useState(0.0);
  const [scTorque, setScTorque] = useState(0.0);
  const [scWear, setScWear] = useState(0.0);
  const [forecastResult, setForecastResult] = useState(null);

  // ── TAB 3: REGISTRY STATES ──────────────────────────────────
  const [assets, setAssets] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [newAsset, setNewAsset] = useState({
    id: '',
    type: 'Custom',
    location: 'Line A - Cell 2',
    description: 'Main Extruder Spindle Motor',
    sensors: {
      rpm: 1500,
      torque: 40,
      air_temp: 300,
      proc_temp: 310,
      tool_wear: 50
    }
  });

  // ── TAB 4: BATCH ANALYZER STATES ────────────────────────────
  const [batchData, setBatchData] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [useBuiltin, setUseBuiltin] = useState(false);
  const fileInputRef = useRef(null);

  // ── TAB 5: HEAT EXCHANGERS STATES ───────────────────────────
  const [exchangers, setExchangers] = useState([]);
  const [hxLoading, setHxLoading] = useState(false);

  // ─────────────────────────────────────────────
  // LOAD INITIAL CONFIG & THRESHOLDS
  // ─────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then(res => res.json())
      .then(data => {
        setConfig(data);
        if (data.models_loaded) {
          setActiveThresh(data.thresholds.xgb_f1);
        }
      })
      .catch(err => console.error("Error loading config:", err));
  }, []);

  // Sync threshold with operational mode preset
  useEffect(() => {
    if (!config) return;
    if (opMode === 'Balanced') {
      setActiveThresh(config.thresholds.xgb_f1);
    } else {
      setActiveThresh(config.thresholds.xgb_safety);
    }
  }, [opMode, config]);

  // ─────────────────────────────────────────────
  // RUN SIMULATOR inference
  // ─────────────────────────────────────────────
  useEffect(() => {
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          air_temp: simAir,
          proc_temp: simProc,
          rpm: simRpm,
          torque: simTorque,
          tool_wear: simWear,
          type_str: simType,
          active_thresh: activeThresh
        })
      })
      .then(res => res.json())
      .then(data => setSimResult(data))
      .catch(err => console.error("Error running simulation:", err));
    }, 250); // debounce API calls
    return () => clearTimeout(timer);
  }, [simAir, simProc, simRpm, simTorque, simWear, simType, activeThresh]);

  // ─────────────────────────────────────────────
  // RUN FORECAST engine
  // ─────────────────────────────────────────────
  useEffect(() => {
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          air_temp: foreAir,
          proc_temp: foreProc,
          rpm: foreRpm,
          torque: foreTorque,
          tool_wear: foreWear,
          type_str: foreType,
          active_thresh: activeThresh,
          horizon_months: foreHorizon,
          wear_rate: foreWearRate,
          rpm_drift: foreRpmDrift,
          torque_drift: foreTorqDrift,
          sc_rpm: scRpm,
          sc_torque: scTorque,
          sc_wear: scWear
        })
      })
      .then(res => res.json())
      .then(data => setForecastResult(data))
      .catch(err => console.error("Error running forecast:", err));
    }, 300);
    return () => clearTimeout(timer);
  }, [foreAir, foreProc, foreRpm, foreTorque, foreWear, foreType, activeThresh,
      foreHorizon, foreWearRate, foreRpmDrift, foreTorqDrift, scRpm, scTorque, scWear]);

  // ─────────────────────────────────────────────
  // LOAD ASSETS REGISTRY
  // ─────────────────────────────────────────────
  const loadAssets = () => {
    fetch(`${API_BASE}/assets`)
      .then(res => res.json())
      .then(data => setAssets(data))
      .catch(err => console.error("Error loading assets:", err));
  };

  useEffect(() => {
    loadAssets();
  }, []);

  const handleAddAsset = (e) => {
    e.preventDefault();
    fetch(`${API_BASE}/assets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newAsset)
    })
    .then(() => {
      setShowAddModal(false);
      loadAssets();
    })
    .catch(err => console.error("Error adding asset:", err));
  };

  const handleDeleteAsset = (id) => {
    if (window.confirm(`Delete asset ${id}?`)) {
      fetch(`${API_BASE}/assets/${id}`, { method: 'DELETE' })
        .then(() => loadAssets())
        .catch(err => console.error("Error deleting asset:", err));
    }
  };

  // ─────────────────────────────────────────────
  // LOAD HEAT EXCHANGERS
  // ─────────────────────────────────────────────
  const loadExchangers = () => {
    setHxLoading(true);
    fetch(`${API_BASE}/exchangers`)
      .then(res => res.json())
      .then(data => {
        setExchangers(data);
        setHxLoading(false);
      })
      .catch(err => {
        console.error("Error loading exchangers:", err);
        setHxLoading(false);
      });
  };

  useEffect(() => {
    if (activeTab === 'exchanger') {
      loadExchangers();
    }
  }, [activeTab]);

  // ─────────────────────────────────────────────
  // BATCH processing
  // ─────────────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    processBatchFile(file);
  };

  const processBatchFile = (file) => {
    setBatchLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('active_thresh', activeThresh.toString());

    fetch(`${API_BASE}/batch`, {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      setBatchData(data);
      setBatchLoading(false);
    })
    .catch(err => {
      alert("Failed processing batch file");
      setBatchLoading(false);
    });
  };

  // Load builtin data simulation
  useEffect(() => {
    if (useBuiltin) {
      // Simulate file download & trigger API
      setBatchLoading(true);
      fetch('/ai4i2020.csv')
        .then(res => {
          if (!res.ok) throw new Error("Builtin CSV missing in public assets");
          return res.blob();
        })
        .then(blob => {
          const file = new File([blob], 'ai4i2020.csv', { type: 'text/csv' });
          processBatchFile(file);
        })
        .catch(err => {
          // Fallback to fetch from root path or simulate
          setUseBuiltin(false);
          setBatchLoading(false);
          alert("Built-in data is only available if ai4i2020.csv is in your React public/ folder.");
        });
    }
  }, [useBuiltin]);

  // ─────────────────────────────────────────────
  // INTERACTIVE PLOTS & RENDER
  // ─────────────────────────────────────────────
  const getLineData = () => {
    if (!forecastResult || !forecastResult.base) return [];
    return forecastResult.base.months.map((m, i) => {
      const baseProb = forecastResult.base.probabilities[i] * 100;
      const scenProb = forecastResult.scenario ? forecastResult.scenario.probabilities[i] * 100 : null;
      return {
        month: `M ${m}`,
        "Base Trajectory (%)": baseProb,
        "Override Scenario (%)": scenProb
      };
    });
  };

  const lineData = getLineData();

  return (
    <div className="layout-with-sidebar">
      {/* ── SIDEBAR PANEL ── */}
      <div className="sidebar-panel">
        <h3 style={{ marginBottom: '8px', color: '#fff', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Cpu className="pulse-green" size={20} /> Calibration Center
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
          Calibrate alert thresholds for the facility's risk posture.
        </p>
        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '14px 0' }} />
        
        <div className="form-group">
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
            Operational Posture
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input 
                type="radio" 
                name="opMode" 
                value="Balanced" 
                checked={opMode === 'Balanced'} 
                onChange={() => setOpMode('Balanced')}
              />
              ⚖️ Balanced Mode (Optimal F1)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input 
                type="radio" 
                name="opMode" 
                value="Safety" 
                checked={opMode === 'Safety'} 
                onChange={() => setOpMode('Safety')}
              />
              🛡️ Safety-First (90% Recall)
            </label>
          </div>
        </div>

        <div className={`preset-banner ${opMode === 'Balanced' ? 'balanced' : 'safety'}`}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase' }}>
            {opMode} Focus Preset
          </span>
          <div style={{ fontSize: '0.9rem', marginTop: '4px', fontWeight: 600 }}>
            Alarm Limit: {(activeThresh * 100).toFixed(0)}%
          </div>
        </div>

        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
        <h4 style={{ color: '#fff', fontSize: '0.9rem', marginBottom: '8px' }}>📡 System Status</h4>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span>Primary Model: RandomForest (Constrained)</span>
          <span>Scaler Matrix: standard_scaler</span>
          <span>Sub-models: 5 Multi-Output Units</span>
        </div>
        <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
        <span style={{ color: 'var(--text-dark)', fontSize: '0.75rem' }}>Confidential. Internal plant monitoring systems only.</span>
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div className="main-content">
        <header className="app-header">
          <div className="header-title">
            <h1>🏭 predictive_maint_system_v3.0</h1>
            <p>
              <span className="pulse-indicator"></span> 
              STATUS: ONLINE &nbsp;|&nbsp; 
              ENGINE: RandomForest (Constrained) &nbsp;|&nbsp; 
              ACTIVE THRESHOLD: {(activeThresh * 100).toFixed(0)}%
            </p>
          </div>
        </header>

        {/* ── TABS SELECTOR ── */}
        <div className="tabs-container">
          <button className={`tab-button ${activeTab === 'simulator' ? 'active' : ''}`} onClick={() => setActiveTab('simulator')}>
            <Activity size={16} /> Digital Twin Simulator
          </button>
          <button className={`tab-button ${activeTab === 'forecast' ? 'active' : ''}`} onClick={() => setActiveTab('forecast')}>
            <Clock size={16} /> Time-to-Failure Forecast
          </button>
          <button className={`tab-button ${activeTab === 'registry' ? 'active' : ''}`} onClick={() => { setActiveTab('registry'); loadAssets(); }}>
            <Database size={16} /> Plant Asset Registry
          </button>
          <button className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>
            <FileText size={16} /> Fleet Batch Analyzer
          </button>
          <button className={`tab-button ${activeTab === 'exchanger' ? 'active' : ''}`} onClick={() => { setActiveTab('exchanger'); loadExchangers(); }}>
            <Thermometer size={16} /> Heat Exchanger Monitor
          </button>
        </div>

        {/* ─────────────────────────────────────────────
            TAB 1: DIGITAL TWIN SIMULATOR
           ───────────────────────────────────────────── */}
        {activeTab === 'simulator' && simResult && (
          <div>
            <div className="section-header">⚙️ Telemetry Controls & AI Assessment</div>
            <div className="grid-2">
              {/* Telemetry overrides */}
              <div className="glass-card">
                <h4>🎛️ Live Parameter Override</h4>
                
                <div className="form-group">
                  <label>Component Grade Variant</label>
                  <select className="form-select" value={simType} onChange={(e) => setSimType(e.target.value)}>
                    <option>L (Low)</option>
                    <option>M (Medium)</option>
                    <option>H (High)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Outside Intake Temperature (K)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="295" max="305" step="0.1" value={simAir} onChange={(e) => setSimAir(parseFloat(e.target.value))} />
                    <span className="slider-value">{simAir.toFixed(1)} K</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Internal Chamber Temperature (K)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="305" max="315" step="0.1" value={simProc} onChange={(e) => setSimProc(parseFloat(e.target.value))} />
                    <span className="slider-value">{simProc.toFixed(1)} K</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Agitator / Rotational Speed (RPM)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="1168" max="2886" step="10" value={simRpm} onChange={(e) => setSimRpm(parseInt(e.target.value))} />
                    <span className="slider-value">{simRpm} RPM</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Torque Output Load (Nm)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="3.8" max="76.6" step="0.1" value={simTorque} onChange={(e) => setSimTorque(parseFloat(e.target.value))} />
                    <span className="slider-value">{simTorque.toFixed(1)} Nm</span>
                  </div>
                </div>

                <div className="form-group">
                  <label>Component Run-Time Wear (min)</label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="0" max="253" step="1" value={simWear} onChange={(e) => setSimWear(parseInt(e.target.value))} />
                    <span className="slider-value">{simWear} min</span>
                  </div>
                </div>

                <hr style={{ border: '0', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '20px 0' }} />
                <h5 style={{ marginBottom: '12px', fontSize: '0.9rem', color: '#fff' }}>🧪 Stress Load Test Profiles</h5>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn btn-danger" style={{ flexGrow: 1 }} onClick={() => { setSimAir(298.5); setSimProc(307.0); setSimRpm(1340); setSimTorque(48.0); setSimWear(50); setSimType("L (Low)"); }}>
                    🔴 Heat Lock (HDF)
                  </button>
                  <button className="btn" style={{ flexGrow: 1, borderColor: '#f59e0b', color: '#fbbf24' }} onClick={() => { setSimAir(300.0); setSimProc(310.0); setSimRpm(2700); setSimTorque(10.0); setSimWear(80); setSimType("M (Medium)"); }}>
                    🟡 High Power (PWF)
                  </button>
                  <button className="btn" style={{ flexGrow: 1, borderColor: '#10b981', color: '#34d399' }} onClick={() => { setSimAir(300.0); setSimProc(310.0); setSimRpm(1550); setSimTorque(38.0); setSimWear(80); setSimType("H (High)"); }}>
                    🟢 Stable Median
                  </button>
                </div>
              </div>

              {/* Assessment ring and status */}
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <h4>AI Assessment Output</h4>
                  <Gauge prob={simResult.probability} activeThresh={activeThresh} />
                </div>

                <div style={{ marginTop: '16px' }}>
                  {simResult.probability >= activeThresh ? (
                    <div className="alert-critical">
                      <h2>🚨 HIGH FAILURE SHUTDOWN RISK 🚨</h2>
                      <p className="alert-desc">Anomalous load patterns exceed safe operating limit of {(activeThresh * 100).toFixed(0)}%</p>
                    </div>
                  ) : simResult.probability >= 0.35 ? (
                    <div className="alert-warning">
                      <h2>⚠️ MAINTENANCE ADVISORY</h2>
                      <p className="alert-desc">Operating threshold is warning ({(simResult.probability * 100).toFixed(1)}%). Inspections required.</p>
                    </div>
                  ) : (
                    <div className="alert-safe">
                      <h2>🟢 COMPONENT HEALTH STABLE</h2>
                      <p className="alert-desc">Nominal risk factor ({(simResult.probability * 100).toFixed(1)}%)</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Computed signals */}
            <div className="section-header">📐 Computed Operational Signals</div>
            <div className="grid-4">
              <div className="metric-card border-top" style={{ borderTopColor: '#0284c7' }}>
                <div className="metric-title">Thermodynamic Delta</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.temp_delta.toFixed(2)}</span>
                  <span className="metric-unit">Kelvin (K)</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: simResult.power_W >= 3500 && simResult.power_W <= 9000 ? '#22c55e' : '#ef4444' }}>
                <div className="metric-title">Rotational Power</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.power_W.toFixed(0)}</span>
                  <span className="metric-unit">Watts (W)</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: '#8b5cf6' }}>
                <div className="metric-title">Overstrain Index</div>
                <div className="metric-value-container">
                  <span className="metric-value">{simResult.torque_x_wear.toFixed(0)}</span>
                  <span className="metric-unit">N-m-Min</span>
                </div>
              </div>
              <div className="metric-card border-top" style={{ borderTopColor: '#14b8a6' }}>
                <div className="metric-title">Life Wear Factor</div>
                <div className="metric-value-container">
                  <span className="metric-value">{(simResult.wear_pct * 100).toFixed(1)}</span>
                  <span className="metric-unit">Percent (%)</span>
                </div>
              </div>
            </div>

            {/* Deviations comparison */}
            <div className="section-header">📈 Parameter Deviation Analysis (Compared to Healthy Fleet Medians)</div>
            <div className="grid-2">
              <div className="glass-card">
                {simResult.deviations.map((dev) => (
                  <div key={dev.parameter} style={{ margin: '14px 0' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                      <span>{dev.parameter}</span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {dev.current} vs {dev.median} ({dev.deviation >= 0 ? '+' : ''}{dev.deviation}%)
                      </span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                      <div style={{
                        position: 'absolute',
                        left: '50%',
                        width: `${Math.min(50, Math.abs(dev.deviation) / 2)}%`,
                        transform: dev.deviation < 0 ? 'translateX(-100%)' : 'none',
                        height: '100%',
                        background: Math.abs(dev.deviation) > 20 ? '#ef4444' : (Math.abs(dev.deviation) > 10 ? '#f59e0b' : '#10b981'),
                        boxShadow: `0 0 8px ${Math.abs(dev.deviation) > 20 ? 'rgba(239,68,68,0.5)' : (Math.abs(dev.deviation) > 10 ? 'rgba(245,158,11,0.5)' : 'rgba(16,185,129,0.5)')}`,
                        borderRadius: '2px'
                      }}></div>
                      <div style={{ position: 'absolute', left: '50%', width: '1px', height: '100%', background: '#fff', opacity: 0.3 }}></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="glass-card">
                <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '14px' }}>📊 Out-of-Spec Diagnostic Flags</h5>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.9rem' }}>
                  {simResult.deviations.filter(d => Math.abs(d.deviation) > 10).length > 0 ? (
                    simResult.deviations.map((dev) => {
                      if (Math.abs(dev.deviation) > 20) {
                        return (
                          <div key={dev.parameter} style={{ borderLeft: '3px solid #ef4444', paddingLeft: '12px', margin: '4px 0' }}>
                            <span style={{ color: '#f87171', fontWeight: 600 }}>🛑 Critical Deviation:</span> <strong>{dev.parameter}</strong> deviates by {dev.deviation > 0 ? '+' : ''}{dev.deviation}% from baseline median.
                          </div>
                        );
                      } else if (Math.abs(dev.deviation) > 10) {
                        return (
                          <div key={dev.parameter} style={{ borderLeft: '3px solid #f59e0b', paddingLeft: '12px', margin: '4px 0' }}>
                            <span style={{ color: '#fbbf24', fontWeight: 600 }}>⚠️ Warning Deviation:</span> <strong>{dev.parameter}</strong> deviates by {dev.deviation > 0 ? '+' : ''}{dev.deviation}% from baseline.
                          </div>
                        );
                      }
                      return null;
                    })
                  ) : (
                    <div style={{ color: 'var(--text-safe)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={16} color="#10b981" /> All telemetry signals match historical healthy fleet baselines.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Failure modes sub models */}
            <div className="section-header">🤖 Failure Mode Probabilities (Sub-Model Outputs)</div>
            <div className="grid-2">
              <div className="glass-card">
                {Object.entries(simResult.sub_probs).map(([mode, prob]) => {
                  const pct = prob * 100;
                  return (
                    <div key={mode} style={{ margin: '14px 0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                        <span>{mode} Classifier</span>
                        <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{pct.toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: pct >= 35 ? 'linear-gradient(90deg, #b91c1c, #ef4444)' : 'linear-gradient(90deg, #1e293b, #334155)',
                          boxShadow: pct >= 35 ? '0 0 8px rgba(239, 68, 68, 0.4)' : 'none',
                          borderRadius: '4px',
                          transition: 'width 0.6s ease'
                        }}></div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '12px' }}>⚡ Sub-Model Diagnosis</h5>
                  {Object.entries(simResult.sub_probs).some(([_, p]) => p >= 0.35) ? (
                    (() => {
                      const highestMode = Object.entries(simResult.sub_probs).sort((a,b) => b[1] - a[1])[0];
                      return (
                        <div style={{ borderLeft: '3px solid #ef4444', paddingLeft: '14px' }}>
                          <p style={{ fontWeight: 600, color: '#f87171', fontSize: '0.95rem' }}>AI predicts elevated risk of {highestMode[0]} ({(highestMode[1]*100).toFixed(1)}%)</p>
                          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                            Ensure corresponding coolant feeds, lubrication systems, and torque loads are inspected.
                          </p>
                        </div>
                      );
                    })()
                  ) : (
                    <div style={{ color: 'var(--text-safe)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={16} color="#10b981" /> Stable state. All target failure mode sub-model probabilities are below warning limits.
                    </div>
                  )}
                </div>

                <div style={{ marginTop: '20px' }}>
                  <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '8px' }}>💡 Facility Operator Action Guidelines</h5>
                  {simResult.guidelines.map((g, i) => (
                    <div key={i} className="insight-box" style={{ margin: '6px 0', fontSize: '0.8rem' }}>
                      {g.replace(/\*\*/g, '')}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─────────────────────────────────────────────
            TAB 2: TIME-TO-FAILURE FORECAST
           ───────────────────────────────────────────── */}
        {activeTab === 'forecast' && (
          <div>
            <div className="section-header">🕐 Time-to-Failure Forecasting Engine</div>
            <div className="grid-2">
              {/* Forecast controls */}
              <div className="glass-card">
                <h4>⚙️ Machine Parameters & Trajectory Drifts</h4>
                
                <div className="form-group">
                  <label>Component Grade</label>
                  <select className="form-select" value={foreType} onChange={(e) => setForeType(e.target.value)}>
                    <option>L (Low)</option>
                    <option>M (Medium)</option>
                    <option>H (High)</option>
                  </select>
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Air Temperature (K)</label>
                    <input type="number" className="form-input" value={foreAir} step="0.5" onChange={(e) => setForeAir(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Process Temperature (K)</label>
                    <input type="number" className="form-input" value={foreProc} step="0.5" onChange={(e) => setForeProc(parseFloat(e.target.value))} />
                  </div>
                </div>

                <div className="grid-3">
                  <div className="form-group">
                    <label>Rotational Speed (RPM)</label>
                    <input type="number" className="form-input" value={foreRpm} step="50" onChange={(e) => setForeRpm(parseInt(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Torque (Nm)</label>
                    <input type="number" className="form-input" value={foreTorque} step="1.0" onChange={(e) => setForeTorque(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Current Tool Wear (min)</label>
                    <input type="number" className="form-input" value={foreWear} step="5" onChange={(e) => setForeWear(parseInt(e.target.value))} />
                  </div>
                </div>

                <div className="form-group">
                  <label>Forecast Horizon: <strong>{foreHorizon} months</strong></label>
                  <div className="slider-container">
                    <input type="range" className="slider-input" min="3" max="24" step="1" value={foreHorizon} onChange={(e) => setForeHorizon(parseInt(e.target.value))} />
                  </div>
                </div>

                <h5 style={{ marginBottom: '12px', fontSize: '0.9rem', color: '#fff' }}>📉 Degradation Rates (per month)</h5>
                <div className="grid-3">
                  <div className="form-group">
                    <label>Wear rate (min/mo)</label>
                    <input type="number" className="form-input" value={foreWearRate} step="0.5" onChange={(e) => setForeWearRate(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>RPM drift (RPM/mo)</label>
                    <input type="number" className="form-input" value={foreRpmDrift} step="1.0" onChange={(e) => setForeRpmDrift(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Torque drift (Nm/mo)</label>
                    <input type="number" className="form-input" value={foreTorqDrift} step="0.1" onChange={(e) => setForeTorqDrift(parseFloat(e.target.value))} />
                  </div>
                </div>

                <h5 style={{ marginBottom: '6px', fontSize: '0.9rem', color: '#fff' }}>🔭 What-If Scenario Overrides</h5>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>Apply immediate adjustments (e.g. maintenance clean offset) to see curves shift.</p>
                <div className="grid-3">
                  <div className="form-group">
                    <label>RPM adjustment (Δ)</label>
                    <input type="number" className="form-input" value={scRpm} step="50" onChange={(e) => setScRpm(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Torque adjustment (Δ)</label>
                    <input type="number" className="form-input" value={scTorque} step="1.0" onChange={(e) => setScTorque(parseFloat(e.target.value))} />
                  </div>
                  <div className="form-group">
                    <label>Tool wear adjustment (Δ)</label>
                    <input type="number" className="form-input" value={scWear} step="5.0" onChange={(e) => setScWear(parseFloat(e.target.value))} />
                  </div>
                </div>
              </div>

              {/* Forecast chart and status */}
              {forecastResult && forecastResult.base && (
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <h4>Degradation Forecast Trajectory</h4>
                    
                    <div className="grid-3" style={{ marginBottom: '14px' }}>
                      <div className="metric-card">
                        <div className="metric-title">Current Risk</div>
                        <div className="metric-value-container">
                          <span className="metric-value" style={{ color: (forecastResult.base.probabilities[0]*100) >= 35 ? '#f59e0b' : '#10b981' }}>
                            {(forecastResult.base.probabilities[0] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-title">Horizon Risk</div>
                        <div className="metric-value-container">
                          <span className="metric-value" style={{ color: (forecastResult.base.probabilities[forecastResult.base.probabilities.length-1]*100) >= 65 ? '#ef4444' : '#10b981' }}>
                            {(forecastResult.base.probabilities[forecastResult.base.probabilities.length-1] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-title">Predicted Failure</div>
                        <div className="metric-value-container">
                          <span className="metric-value" style={{ color: forecastResult.base.predicted_fail_month ? '#ef4444' : '#10b981' }}>
                            {forecastResult.base.predicted_fail_month ? `Month ${forecastResult.base.predicted_fail_month}` : 'None'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div style={{ width: '100%', height: '240px', background: 'rgba(0,0,0,0.1)', padding: '10px', borderRadius: '10px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={lineData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="month" stroke="var(--text-muted)" fontSize={12} />
                          <YAxis stroke="var(--text-muted)" domain={[0, 100]} fontSize={12} />
                          <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px' }} />
                          <Legend wrapperStyle={{ fontSize: '11px', color: '#fff' }} />
                          <ReferenceLine y={activeThresh * 100} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Alarm Limit', fill: '#ef4444', position: 'top', fontSize: 11 }} />
                          <Line type="monotone" dataKey="Base Trajectory (%)" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                          {forecastResult.scenario && (
                            <Line type="monotone" dataKey="Override Scenario (%)" stroke="#10b981" strokeWidth={2.5} strokeDasharray="5 5" dot={{ r: 3 }} />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div style={{ marginTop: '16px' }}>
                    <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff', marginBottom: '8px' }}>🤖 Predictive TTF Guidance</h5>
                    <div className="insight-box" style={{ borderLeft: '3px solid #3b82f6', fontSize: '0.85rem' }}>
                      {forecastResult.base.fail_message}
                    </div>
                    {forecastResult.base.interventions && forecastResult.base.interventions.length > 0 && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px', paddingLeft: '6px' }}>
                        <strong>🛠️ Recommended Proactive Actions:</strong>
                        {forecastResult.base.interventions.map((inv, idx) => (
                          <div key={idx}>• {inv.intervention} (scheduled for Month {inv.month})</div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─────────────────────────────────────────────
            TAB 3: PLANT ASSET REGISTRY
           ───────────────────────────────────────────── */}
        {activeTab === 'registry' && (
          <div>
            <div className="card-title-flex">
              <div className="section-header" style={{ margin: 0 }}>🏭 Facility Asset Registry</div>
              <div style={{ display: 'flex', gap: '12px' }}>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="🔍 Search Spindle/HX/Pump..." 
                  style={{ width: '240px', padding: '8px 12px' }} 
                  value={searchTerm} 
                  onChange={(e) => setSearchTerm(e.target.value)} 
                />
                <button className="btn btn-primary" onClick={() => setShowAddModal(true)}>
                  <Plus size={16} /> Register Asset
                </button>
              </div>
            </div>

            {/* Asset Add Modal Form (Inline drawer for clean aesthetics) */}
            {showAddModal && (
              <div className="glass-card" style={{ marginBottom: '24px', border: '1px solid var(--color-primary)' }}>
                <h4>Register New Factory Spindle/Component</h4>
                <form onSubmit={handleAddAsset} className="grid-3" style={{ gap: '20px' }}>
                  <div className="form-group">
                    <label>Component Asset ID (e.g., PUMP-105)</label>
                    <input type="text" className="form-input" placeholder="Auto-generated if empty" value={newAsset.id} onChange={(e) => setNewAsset({...newAsset, id: e.target.value})} />
                  </div>
                  <div className="form-group">
                    <label>Machine Type Class</label>
                    <select className="form-select" value={newAsset.type} onChange={(e) => setNewAsset({...newAsset, type: e.target.value})}>
                      <option>Centrifugal Pump</option>
                      <option>Reciprocating Pump</option>
                      <option>Compressor</option>
                      <option>Heat Exchanger</option>
                      <option>Agitator / Mixer</option>
                      <option>Reactor</option>
                      <option>Custom</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Functional Location Code</label>
                    <input type="text" className="form-input" value={newAsset.location} onChange={(e) => setNewAsset({...newAsset, location: e.target.value})} />
                  </div>
                  <div className="form-group" style={{ gridColumn: 'span 3' }}>
                    <label>Description Details</label>
                    <input type="text" className="form-input" value={newAsset.description} onChange={(e) => setNewAsset({...newAsset, description: e.target.value})} />
                  </div>

                  <div style={{ gridColumn: 'span 3', display: 'flex', justifyItems: 'right', gap: '10px' }}>
                    <button type="submit" className="btn btn-primary">Register Asset</button>
                    <button type="button" className="btn" onClick={() => setShowAddModal(false)}>Cancel</button>
                  </div>
                </form>
              </div>
            )}

            {/* Registry table */}
            <div className="table-container">
              <table className="styled-table">
                <thead>
                  <tr>
                    <th>Asset ID</th>
                    <th>Type</th>
                    <th>Location</th>
                    <th>Description</th>
                    <th>Sensor Readings</th>
                    <th>Registered At</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {assets.filter(a => 
                    a.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
                    a.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    a.description.toLowerCase().includes(searchTerm.toLowerCase())
                  ).map((asset) => (
                    <tr key={asset.id}>
                      <td style={{ color: '#fff', fontWeight: 600 }}>{asset.id}</td>
                      <td>
                        <span className="status-pill safe" style={{ background: 'rgba(59,130,246,0.1)', color: '#60a5fa' }}>{asset.type}</span>
                      </td>
                      <td>{asset.location}</td>
                      <td>{asset.description}</td>
                      <td style={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>
                        {asset.sensors ? (
                          Object.entries(asset.sensors).map(([k, v]) => `${k}:${v}`).join(' | ')
                        ) : 'N/A'}
                      </td>
                      <td>{new Date(asset.created_at).toLocaleDateString()}</td>
                      <td>
                        <button className="btn btn-danger" style={{ padding: '4px 8px', borderRadius: '4px' }} onClick={() => handleDeleteAsset(asset.id)}>
                          <Trash2 size={12} />
                        </button>
                      </td>
                    </tr>
                  ))}
                  {assets.length === 0 && (
                    <tr>
                      <td colSpan="7" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                        No assets currently registered in the database.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ─────────────────────────────────────────────
            TAB 4: FLEET BATCH ANALYZER
           ───────────────────────────────────────────── */}
        {activeTab === 'batch' && (
          <div>
            <div className="section-header">📦 Fleet Telemetry Batch Processing</div>
            
            <div className="grid-2">
              <div className="glass-card">
                <h4>CSV Stream Telemetry Uploader</h4>
                <div className="file-upload-area" onClick={() => fileInputRef.current.click()}>
                  <Upload size={32} color="var(--color-primary)" />
                  <div>
                    <span style={{ color: '#fff', fontWeight: 600 }}>Click to browse or drop CSV telemetry export</span>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>Columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]</p>
                  </div>
                  <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept=".csv" onChange={handleFileUpload} />
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input 
                    type="checkbox" 
                    id="builtin_chk" 
                    checked={useBuiltin} 
                    onChange={(e) => setUseBuiltin(e.target.checked)} 
                  />
                  <label htmlFor="builtin_chk" style={{ cursor: 'pointer', fontSize: '0.9rem' }}>📂 Simulate Live Factory Stream (Analyze all 10,000 active records)</label>
                </div>
              </div>

              {batchData && batchData.stats && (
                <div className="glass-card">
                  <h4>Batch Process Results</h4>
                  <div className="grid-5" style={{ gap: '10px' }}>
                    <div className="metric-card">
                      <div className="metric-title">Fleet Units</div>
                      <div className="metric-value-container">
                        <span className="metric-value">{batchData.stats.total}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #ef4444' }}>
                      <div className="metric-title">Critical Risks</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#ef4444' }}>{batchData.stats.critical}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #f59e0b' }}>
                      <div className="metric-title">Warning Risks</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#f59e0b' }}>{batchData.stats.warning}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #10b981' }}>
                      <div className="metric-title">Stable Units</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#10b981' }}>{batchData.stats.safe}</span>
                      </div>
                    </div>
                    <div className="metric-card" style={{ borderTop: '3px solid #3b82f6' }}>
                      <div className="metric-title">Triggered Alarms</div>
                      <div className="metric-value-container">
                        <span className="metric-value" style={{ color: '#60a5fa' }}>{batchData.stats.alarms}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Live accuracy metrics section */}
            {batchData && batchData.evaluation && (
              <div>
                <div className="section-header">📐 Model Accuracy on Your Uploaded Data</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '14px', marginTop: '-6px' }}>
                  Metrics computed against the <strong>Machine failure</strong> ground-truth labels in your file — not the training dataset.
                </p>
                <div className="grid-7">
                  <div className="metric-card border-top" style={{ borderTopColor: '#3b82f6' }}>
                    <div className="metric-title">Accuracy</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{(batchData.evaluation.accuracy * 100).toFixed(2)}</span>
                      <span className="metric-unit">%</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.precision >= 0.5 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">Precision</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.precision.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.recall >= 0.7 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">Recall</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.recall.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.f1_score >= 0.5 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">F1-Score</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.f1_score.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: batchData.evaluation.pr_auc >= 0.6 ? '#10b981' : '#f59e0b' }}>
                    <div className="metric-title">PR-AUC</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.pr_auc.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: '#f59e0b' }}>
                    <div className="metric-title">False Pos</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.false_positives}</span>
                      <span className="metric-unit">rows</span>
                    </div>
                  </div>
                  <div className="metric-card border-top" style={{ borderTopColor: '#ef4444' }}>
                    <div className="metric-title">False Neg</div>
                    <div className="metric-value-container">
                      <span className="metric-value">{batchData.evaluation.false_negatives}</span>
                      <span className="metric-unit">rows</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Process list table */}
            {batchData && batchData.records && (
              <div>
                <div className="section-header">🚨 Fleet Risk Status & Warning List</div>
                <div className="table-container" style={{ maxHeight: '400px' }}>
                  <table className="styled-table">
                    <thead>
                      <tr>
                        <th>UDI</th>
                        <th>Type</th>
                        <th>Air Temp [K]</th>
                        <th>Process Temp [K]</th>
                        <th>Rotational Speed [rpm]</th>
                        <th>Torque [Nm]</th>
                        <th>Tool Wear [min]</th>
                        <th>Failure Probability (%)</th>
                        <th>Risk Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchData.records.map((r, idx) => (
                        <tr key={idx}>
                          <td style={{ color: '#fff', fontWeight: 600 }}>{r.UDI}</td>
                          <td>{r.Type}</td>
                          <td>{r["Air temperature [K]"]}</td>
                          <td>{r["Process temperature [K]"]}</td>
                          <td>{r["Rotational speed [rpm]"]}</td>
                          <td>{r["Torque [Nm]"]}</td>
                          <td>{r["Tool wear [min]"]}</td>
                          <td style={{ fontWeight: 600, color: '#fff' }}>{r["Failure Probability (%)"]}%</td>
                          <td>
                            <span className={`status-pill ${
                              r["Risk Level"].includes('CRITICAL') ? 'critical' : (r["Risk Level"].includes('WARNING') ? 'warning' : 'safe')
                            }`}>
                              {r["Risk Level"].split(' ').slice(1).join(' ')}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {batchLoading && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '10px' }}>
                <RefreshCw size={24} className="pulse-green" style={{ animation: 'spin 1.5s infinite linear' }} />
                <span>Running batch analysis over fleet logs...</span>
                <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
              </div>
            )}
          </div>
        )}

        {/* ─────────────────────────────────────────────
            TAB 5: HEAT EXCHANGERS MONITOR
           ───────────────────────────────────────────── */}
        {activeTab === 'exchanger' && (
          <div>
            <div className="card-title-flex">
              <div className="section-header" style={{ margin: 0 }}>🌡️ Heat Exchanger Performance Fleet Rankings</div>
              <button className="btn btn-primary" onClick={loadExchangers} disabled={hxLoading}>
                <RefreshCw size={14} style={{ animation: hxLoading ? 'spin 1.5s infinite linear' : 'none' }} /> Refresh Fleet Stats
              </button>
            </div>

            <div className="grid-2" style={{ marginTop: '20px' }}>
              {exchangers.map((hx) => (
                <div key={hx.id} className="glass-card" style={{ borderTop: `4px solid ${hx.status_color}`, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '14px' }}>
                      <div>
                        <h4 style={{ margin: 0, border: 'none', padding: 0 }}>{hx.id}</h4>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{hx.description}</span>
                      </div>
                      <span className={`status-pill ${hx.status_label.toLowerCase()}`}>{hx.status_label}</span>
                    </div>

                    <div className="grid-3" style={{ marginBottom: '18px', gap: '10px' }}>
                      <div className="metric-card" style={{ padding: '12px' }}>
                        <div className="metric-title" style={{ fontSize: '0.65rem' }}>Effectiveness</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
                          {(hx.effectiveness * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Design: {(hx.design_effectiveness*100).toFixed(0)}%</div>
                      </div>
                      <div className="metric-card" style={{ padding: '12px' }}>
                        <div className="metric-title" style={{ fontSize: '0.65rem' }}>Fouling Limit</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
                          {hx.foul_pct.toFixed(0)}%
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Rf: {hx.fouling_factor.toFixed(6)}</div>
                      </div>
                      <div className="metric-card" style={{ padding: '12px' }}>
                        <div className="metric-title" style={{ fontSize: '0.65rem' }}>Pinch Duty</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
                          {hx.heat_duty_kw.toFixed(0)} kW
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>U: {hx.current_U.toFixed(0)} W/m²K</div>
                      </div>
                    </div>

                    {/* Progress Bar for Fouling Limit */}
                    <div style={{ margin: '14px 0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                        <span>Fouling Factor Accumulation</span>
                        <span style={{ fontWeight: 600, color: hx.foul_pct >= 80 ? '#ef4444' : '#fff' }}>{hx.foul_pct.toFixed(1)}% Limit</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                          width: `${Math.min(100, hx.foul_pct)}%`,
                          height: '100%',
                          background: hx.foul_pct >= 80 ? '#ef4444' : (hx.foul_pct >= 50 ? '#f59e0b' : '#10b981'),
                          boxShadow: `0 0 8px ${hx.foul_pct >= 80 ? 'rgba(239,68,68,0.4)' : 'transparent'}`,
                          borderRadius: '4px',
                          transition: 'width 0.6s ease'
                        }}></div>
                      </div>
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px', marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontSize: '0.85rem' }}>
                      ⏱️ Days until cleaning required: <strong style={{ color: hx.days_to_clean <= 30 ? '#ef4444' : '#fff' }}>{hx.days_to_clean} days</strong>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Criticality score: <strong style={{ color: hx.status_color }}>{hx.criticality_score.toFixed(0)}</strong>
                    </div>
                  </div>
                </div>
              ))}
              {exchangers.length === 0 && !hxLoading && (
                <div style={{ gridColumn: 'span 2', textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                  No heat exchangers registered or analyzed yet.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
