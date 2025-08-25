// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import ParticlesComponent from './ParticlesBackground';

function App() {
  const [sentence1, setSentence1] = useState('');
  const [sentence2, setSentence2] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    try { //the backend will 'repond' to the request 'posted' by the frontend 
      //which will be stored in 'response' variable
      const response = await axios.post('https://similarity-detector-project-1.onrender.com', {
        sentence1,
        sentence2,
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      alert('Failed to get response from backend');
    }
  };

  return (
    <div className="App">
      <ParticlesComponent id="particles" />
      <div className="container">
        <h1>Text Similarity</h1>
        <textarea
          maxLength={100}
          value={sentence1}
          onChange={(e) => setSentence1(e.target.value)}
          placeholder="Sentence 1 (100 characters max)"
        />
        <textarea
          maxLength={100}
          value={sentence2}
          onChange={(e) => setSentence2(e.target.value)}
          placeholder="Sentence 2 (100 characters max)"
        />
        <button onClick={handleSubmit}>Generate</button>

        {result && (
          <div className="result">
            <p><strong>Sentence 1:</strong> {result.sentence1}</p>
            <p><strong>Sentence 2:</strong> {result.sentence2}</p>
            
            <h3>🔍 Predictions:</h3>
            <ul>
              {Object.entries(result.predictions).map(([model, label]) => (
                <li key={model}><strong>{model}:</strong> {label}</li>
              ))}
            </ul>

            {/* ✅ New Analysis Section */}
            {result.analysis && (
              <div className="analysis">
                <h3>📝 Analysis:</h3>
                <p><strong>Common Words:</strong> {result.analysis.common_words.join(", ") || "NONE"}</p>
                <p><strong>Unique to Sentence 1:</strong> {result.analysis.unique_to_s1.join(", ") || "NONE"}</p>
                <p><strong>Unique to Sentence 2:</strong> {result.analysis.unique_to_s2.join(", ") || "NONE"}</p>
              </div>
            )}

            {result.final_score && (
              <div className="final-score">
                <h3>🏁 Final Score:</h3>
                <p><strong>Plagiarism Percent:</strong> {result.final_score.plagiarism_percent || "NONE"}</p>
                <p><strong>Final Label:</strong> {result.final_score.final_label || "NONE"}</p>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="creators">
        <h5>
          Project by Deepankar Krishna in collaboration with S.aditya Mukherjee
        </h5>
      </div>
    </div>
  );
}

export default App;
