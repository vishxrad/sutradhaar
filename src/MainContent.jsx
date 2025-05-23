import { useState } from "react";
import "./MainContent.css";

function MainContent() {
  const [topic, setTopic] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = () => {
    if (!topic.trim()) return;
    setIsGenerating(true);
    // Simulate generation process
    setTimeout(() => {
      setIsGenerating(false);
      // Here you would integrate with your backend
    }, 3000);
  };

  return (
    <div className="main-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="app-title">सूत्रधार</h1>
          <p className="app-subtitle">Topic to 2-Minute Video Generator</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-wrapper">
          {/* Hero Section */}
          <section className="hero-section">
            <h2 className="hero-title">Transform Any Topic Into Engaging Videos</h2>
            <p className="hero-description">
              Enter a topic and watch as सूत्रधार creates a professional 2-minute video 
              with slides, narration, and visual elements.
            </p>
          </section>

          {/* Input Section */}
          <section className="input-section">
            <div className="input-container">
              <label htmlFor="topic-input" className="input-label">
                What topic would you like to create a video about?
              </label>
              <div className="input-wrapper">
                <input
                  id="topic-input"
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., Climate Change, Machine Learning, Cooking Pasta..."
                  className="topic-input"
                  disabled={isGenerating}
                />
                <button
                  onClick={handleGenerate}
                  disabled={!topic.trim() || isGenerating}
                  className={`generate-btn ${isGenerating ? 'generating' : ''}`}
                >
                  {isGenerating ? (
                    <>
                      <span className="spinner"></span>
                      Generating...
                    </>
                  ) : (
                    'Generate Video'
                  )}
                </button>
              </div>
            </div>
          </section>

          {/* Features Section */}
          {/* <section className="features-section">
            <h3 className="features-title">What You Get</h3>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon">🎬</div>
                <h4>Professional Videos</h4>
                <p>High-quality 2-minute videos with smooth transitions and animations</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📊</div>
                <h4>Auto-Generated Slides</h4>
                <p>Beautiful slides created automatically from your topic with relevant content</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🎙️</div>
                <h4>Voice Narration</h4>
                <p>Natural-sounding voice narration that explains your topic clearly</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🖼️</div>
                <h4>Visual Elements</h4>
                <p>Relevant images and graphics to make your content more engaging</p>
              </div>
            </div>
          </section> */}
        </div>
      </main>
    </div>
  );
}

export default MainContent;
