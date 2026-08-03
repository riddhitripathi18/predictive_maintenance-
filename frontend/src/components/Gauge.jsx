import React from 'react';

const getRisk = (prob) => {
  if (prob > 0.70) return { color: '#ef4444', label: 'CRITICAL' };
  if (prob >= 0.35) return { color: '#f59e0b', label: 'WARNING' };
  return { color: '#10b981', label: 'SAFE' };
};

export default function Gauge({ prob, activeThresh }) {
  const { color, label } = getRisk(prob);
  const r = 80;
  const circ = 2 * Math.PI * r; // ~502.6
  const offset = circ - (prob * circ);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      height: '260px',
      marginTop: '10px'
    }}>
      <style>{`
        @keyframes pulse-ring {
          0% { opacity: 0.8; filter: drop-shadow(0px 0px 4px ${color}); }
          50% { opacity: 1; filter: drop-shadow(0px 0px 16px ${color}); }
          100% { opacity: 0.8; filter: drop-shadow(0px 0px 4px ${color}); }
        }
        .glowing-ring {
          animation: pulse-ring 2s infinite ease-in-out;
        }
      `}</style>
      
      <svg width="220" height="220" viewBox="0 0 200 200" style={{ transform: 'rotate(-90deg)' }}>
        {/* Background base ring */}
        <circle cx="100" cy="100" r={r} stroke="rgba(255,255,255,0.03)" strokeWidth="12" fill="transparent" />
        
        {/* Segment boundaries */}
        <circle cx="100" cy="100" r={r} stroke="#1e293b" strokeWidth="14" fill="transparent" strokeDasharray={circ} strokeDashoffset={0} />
        
        {/* Active Fill Ring */}
        <circle 
          className="glowing-ring" 
          cx="100" 
          cy="100" 
          r={r} 
          stroke={color} 
          strokeWidth="12" 
          fill="transparent"
          strokeDasharray={circ} 
          strokeDashoffset={offset} 
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.8s ease-in-out' }} 
        />
      </svg>
      
      <div style={{
        position: 'absolute',
        textAlign: 'center',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)'
      }}>
        <h1 style={{
          margin: 0,
          fontSize: '2.8rem',
          fontWeight: 700,
          color: '#ffffff',
          textShadow: `0 0 10px ${color}aa`,
          fontFamily: 'Outfit, sans-serif'
        }}>
          {(prob * 100).toFixed(1)}%
        </h1>
        <p style={{
          margin: '2px 0 0 0',
          fontSize: '0.85rem',
          fontWeight: 600,
          color: color,
          textTransform: 'uppercase',
          letterSpacing: '2px'
        }}>
          {label}
        </p>
      </div>
    </div>
  );
}
