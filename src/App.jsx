// import { useState } from "react";
// import SplashScreen from "./SplashScreen.js";
// import MainContent from "./MainContent.jsx";

// function App() {
//   const [showSplash, setShowSplash] = useState(true);
//   const [isTransitioning, setIsTransitioning] = useState(false);

//   const handleStartClick = () => {
//     setIsTransitioning(true);
//     // Wait for fade out animation to complete before switching content
//     setTimeout(() => {
//       setShowSplash(false);
//       setIsTransitioning(false);
//     }, 500); // 500ms matches the CSS transition duration
//   };

//   return (
//     <div className="min-h-screen">
//       <div className={`transition-container ${isTransitioning ? 'fade-out' : 'fade-in'}`}>
//         {showSplash ? <SplashScreen onStartClick={handleStartClick} /> : <MainContent />}
//       </div>
//     </div>
//   );
// }

// export default App;
import { useState } from "react";
import SplashScreen from "./SplashScreen";
import MainContent from "./MainContent";
import "./App.css"; // Make sure this includes your CSS

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const handleStartClick = () => {
    setIsTransitioning(true);

    setTimeout(() => {
      setShowSplash(false);
      setIsTransitioning(false);
    }, 500); // Duration should match the CSS fade-out transition
  };

  return (
    <div className="min-h-screen">
      <div className={`transition-container ${isTransitioning ? 'fade-out' : 'fade-in'}`}>
        {showSplash ? <SplashScreen onStartClick={handleStartClick} /> : <MainContent />}
      </div>
    </div>
  );
}

export default App;
