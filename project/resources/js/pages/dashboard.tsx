import AppLayout from '@/layouts/app-layout';
import { Head } from '@inertiajs/react';
import { useState } from 'react';
import { type BreadcrumbItem } from '@/types';
import * as pdfjsLib from 'pdfjs-dist';
import * as mammoth from 'mammoth';

const breadcrumbs: BreadcrumbItem[] = [{ title: 'Dashboard', href: '/dashboard' }];

type Slide = {
  title: string;
  narration: string;
  slide_content: string[];
};

type EditableSlide = {
  title: string;
  narration: string;
  slide_content: string[];
};

export default function Dashboard() {
  const [topic, setTopic] = useState('');
  const [duration, setDuration] = useState<'summary' | 'recap' | 'explainer'>('recap');
  const [selectedDuration, setSelectedDuration] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [step, setStep] = useState<'initial' | 'generated'>('initial');
  const [editableSlides, setEditableSlides] = useState<EditableSlide[]>([]);
  const [scriptId, setScriptId] = useState('');

  const suggestions = [
    'Introduction to Machine Learning',
    'Climate Change and Global Warming',
    'History of Ancient Civilizations',
    'Quantum Physics Basics',
    'Digital Marketing Strategies',
    'Sustainable Energy Solutions',
  ];

  const getDurationsForType = (type: 'summary' | 'recap' | 'explainer') => {
    switch (type) {
      case 'summary':
        return ['1', '2'];
      case 'recap':
        return ['5', '10', '15'];
      case 'explainer':
        return ['25', '30', '60'];
      default:
        return [];
    }
  };

const handleSaveScript = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/api/save-script', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        script_id: scriptId,
        topic,
        raw_script: topic,
        parsed_script: editableSlides,
      }),
    });

    if (!response.ok) {
      throw new Error('Save failed');
    }

    alert('Script saved/updated successfully!');
  } catch (err) {
    console.error(err);
    alert('Failed to save script');
  }
};



  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();

    if (file.type === 'text/plain') {
      reader.onload = () => setTopic(reader.result as string);
      reader.readAsText(file);
    } else if (file.type === 'application/pdf') {
      const readPDF = async (file: File) => {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let text = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const pageText = content.items.map((item: any) => item.str).join(' ');
          text += pageText + '\n';
        }
        setTopic(text);
      };
      readPDF(file);
    } else if (file.name.endsWith('.docx')) {
      reader.onload = async () => {
        const result = await mammoth.extractRawText({ arrayBuffer: reader.result as ArrayBuffer });
        setTopic(result.value);
      };
      reader.readAsArrayBuffer(file);
    } else {
      alert(`Unsupported file type: ${file.name}`);
    }
  };

  const generatePresentation = async () => {
    if (!topic.trim()) {
      alert('Please enter a topic.');
      return;
    }

    if (!selectedDuration) {
      alert('Please select a duration.');
      return;
    }

    setIsGenerating(true);

    try {
      const API_BASE = 'https://sutradhaar.zsapiens.com';
      const customMinutes = parseInt(selectedDuration);

      const scriptRes = await fetch(`${API_BASE}/generate-script`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          video_type: duration,
          custom_duration_minutes: customMinutes,
        }),
      });

      if (!scriptRes.ok) {
        const err = await scriptRes.json();
        throw new Error(err?.detail || 'Script generation failed');
      }

      const scriptData = await scriptRes.json();
      const editable = scriptData.parsed_script.map((slide: Slide) => ({
        title: slide.title,
        narration: slide.narration,
        slide_content: slide.slide_content || [],
      }));

      setScriptId(scriptData.script_id);
      setEditableSlides(editable);
      setStep('generated');
    } catch (err) {
      console.error(err);
      alert('Something went wrong while generating script.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSlideChange = (index: number, key: 'title' | 'narration' | 'slide_content', value: string | string[]) => {
    const updated = [...editableSlides];
    if (key === 'slide_content') {
      updated[index][key] = (value as string).split('\n');
    } else {
      updated[index][key] = value as string;
    }
    setEditableSlides(updated);
  };

  return (
    <AppLayout breadcrumbs={breadcrumbs}>
      <Head title="Dashboard" />
      <div className="flex flex-col items-center justify-center min-h-[80vh] px-6">
        <div className="max-w-4xl w-full">
          {step === 'initial' && (
            <>
              <div className="text-center space-y-6 mb-12">
                <p className="text-xl md:text-2xl text-gray-600 leading-relaxed">
                  AI-Powered Video Presentation Generator
                </p>
                <p className="text-lg text-gray-500">Transform Text to Video</p>
              </div>

              <div className="bg-white/80 backdrop-blur-sm border border-gray-200 rounded-2xl p-8 shadow-xl mb-8 space-y-6">
                {/* Duration type buttons */}
                <div className="flex flex-wrap justify-between items-center gap-4">
                  <div className="flex gap-2">
                    {['summary', 'recap', 'explainer'].map((value) => (
                      <button
                        key={value}
                        onClick={() => {
                          setDuration(value as 'summary' | 'recap' | 'explainer');
                          setSelectedDuration('');
                        }}
                        className={`px-4 py-2 rounded-full text-sm font-medium border transition-colors duration-200 ${
                          duration === value
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100'
                        }`}
                      >
                        {value.charAt(0).toUpperCase() + value.slice(1)}
                      </button>
                    ))}
                  </div>

                  {/* Duration dropdown */}
                  <select
                    className="border border-gray-300 px-3 py-2 rounded-md text-sm text-gray-700"
                    value={selectedDuration}
                    onChange={(e) => setSelectedDuration(e.target.value)}
                  >
                    <option value="">Select Duration</option>
                    {getDurationsForType(duration).map((time) => (
                      <option key={time} value={time}>
                        {time} min
                      </option>
                    ))}
                  </select>
                </div>

                {/* Textarea */}
                <div className="relative">
                  <textarea
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Enter your script or topic here..."
                    rows={4}
                    className="w-full px-4 pr-10 py-3 border border-gray-300 rounded-xl text-base focus:border-blue-500 focus:outline-none shadow-sm resize-none min-h-[48px] max-h-[200px] overflow-y-auto"
                  />
                  <label htmlFor="scriptUpload">
                    <div className="absolute bottom-2.5 right-3 cursor-pointer">
                      <img src="/icon.jpeg" alt="Upload" className="w-5 h-5 opacity-60 hover:opacity-100" />
                    </div>
                  </label>
                  <input
                    type="file"
                    id="scriptUpload"
                    accept=".pdf,.doc,.docx,.txt"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>

                {/* Suggestions */}
                <div>
                  <p className="text-sm text-gray-600 mb-2">Quick suggestions:</p>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((item, i) => (
                      <button
                        key={i}
                        onClick={() => setTopic(item)}
                        className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200"
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={generatePresentation}
                  disabled={isGenerating}
                  className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-4 text-lg rounded-lg transition duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isGenerating ? 'Generating...' : 'Generate Presentation'}
                </button>
              </div>
            </>
          )}

          {step === 'generated' && (
            <div className="w-full max-w-4xl mx-auto bg-white border border-gray-200 rounded-2xl p-6 shadow-xl mt-6 space-y-6">
              <h2 className="text-xl font-bold text-center">Editable Script</h2>
              <div className="max-h-[500px] overflow-y-auto space-y-4">
                {editableSlides.map((slide, index) => (
              <div className="border p-5 rounded-xl shadow-md bg-white transition hover:shadow-lg space-y-4">
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">Slide Title </label>
    <input
      type="text"
      value={slide.title}
      onChange={(e) => handleSlideChange(index, 'title', e.target.value)}
      className="w-full border border-gray-300 rounded-md px-3 py-2 font-semibold text-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50"
      placeholder="Slide Title"
    />
  </div>

  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">Narration (Voiceover Script)</label>
    <textarea
      value={slide.narration}
      onChange={(e) => handleSlideChange(index, 'narration', e.target.value)}
      rows={4}
      className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50"
      placeholder="Enter narration here..."
    />
  </div>

<div>
  <label className="block text-sm font-medium text-purple-700 mb-1">
    On-Screen Content (Bullet Points)
  </label>
<textarea
  value={slide.slide_content.map(point => `• ${point}`).join('\n')}
  onChange={(e) => {
    const lines = e.target.value
      .split('\n')
      .map(line => line.replace(/^•\s*/, '')) // remove bullet prefix
      .filter(line => line.trim() !== '');
    handleSlideChange(index, 'slide_content', lines);
  }}
  rows={4}
  className="w-full border rounded-md px-3 py-2 text-sm bg-gray-50 font-mono leading-6"
  placeholder="• First point\n• Second point"
/>

</div>
</div>

                ))}
              </div>
              <div className="flex justify-between pt-4">
                <button
                  onClick={() => setStep('initial')}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                >
                  ← Back
                </button>
              <button
  onClick={handleSaveScript}
  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
>
  Next →
</button>

              </div>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
