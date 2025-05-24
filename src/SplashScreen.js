import React, { useEffect, useState } from "react";
import "./SplashScreen.css";

const characters = ["सू", "त्र", "धा", "र"];

// const greetings = [
//   "नमस्ते", "नमस्कार", "Hello", "Hola", "Bonjour", "Ciao", "Hallo", "こんにちは",
//   "안녕하세요", "你好", "Olá", "Привет", "مرحبا", "שלום", "สวัสดี", "வணக்கம்",
//   "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "Hej", "Selamat pagi", "Sawubona", "Salve"
// ];
const greetings = [
  "Transform Text to Video ", "पाठ को वीडियो में बदलें", "Trasforma il testo in video", "转文本为视频", "حوّل النص إلى فيديو", "Transformez le texte en vidéo", "ಪಠ್ಯವನ್ನು ವೀಡಿಯೋಗೆ ಪರಿವರ್ತನೆ ಮಾಡಿ",
  "Chuyển đổi văn bản thành video", "पाठं चित्रपटं परिवर्तनं", "पाठ के वीडियो में बदलिए", "Vakavuna na Ivola me Vakaata na Video", "Text in Video umwandeln", "Ubah Teks Menjadi Video", "पाठलाई भिडियामा रूपान्तरण गर्नुहोस्", "ലേഖനം വിഡിയോയുടെ രൂപത്തിലേക്ക് പരിവർത്തനം ചെയ്യുക",
  "मजकूराचें व्हिडियोंत रुपांतर करप", "पाठ के वीडियो में रूपांतरित करू", "Преобразовать текст в видео"
];
function SplashScreen({ onStartClick }) {
  const [animated, setAnimated] = useState(false);
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);

  useEffect(() => {
    setAnimated(true); // Start animation for title

    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => (prev + 1) % greetings.length);
        setFade(true);
      }, 850);
    }, 700);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container">
      <div className="logo">
        <h1 id="mainTitle">
          {characters.map((char, index) => (
            <span
              key={index}
              className={`letter ${animated ? "animate" : ""}`}
              style={{ animationDelay: `${index * 0.3}s` }}
            >
              {char}
            </span>
          ))}
        </h1>
      </div>

      {/* Multilingual greeting between title and button */}
      <p className={`intro-line ${fade ? "fade-in" : "fade-out"}`}>
        {greetings[index]}
      </p>

      <button className="start-button" onClick={onStartClick}>Start</button>
    </div>
  );
}

export default SplashScreen;