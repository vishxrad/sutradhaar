import { useState, useEffect } from "react";
import SplashScreen from "./SplashScreen";
import MainContent from "./MainContent";

function App() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen">
      {showSplash ? <SplashScreen /> : <MainContent />}
    </div>
  );
}

export default App;
