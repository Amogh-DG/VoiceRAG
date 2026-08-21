import { VoiceQA } from './components/VoiceQA';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <div className="ambient-background">
        <div className="ambient-glow glow-1" />
        <div className="ambient-glow glow-2" />
      </div>

      <nav className="app-navbar">
        <div className="nav-logo">
          <span className="nav-icon">🎙️</span>
          <span className="nav-brand">VoiceQA</span>
        </div>
        <div className="nav-tagline">
          <span>AI-Powered Speech Query System</span>
        </div>
      </nav>

      <main className="app-content">
        <VoiceQA />
      </main>

      <footer className="app-footer">
        <p>Powered by Sarvam AI STT &bull; Privacy-first Voice Interface</p>
      </footer>
    </div>
  );
}

export default App;
