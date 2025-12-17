// my-book/src/components/Chatbot.jsx
import React, { useState, useEffect } from 'react';
import styles from './Chatbot.module.css';

// The URL of your FastAPI backend running on http://127.0.0.1:8000
// IMPORTANT: This URL must be correct and your FastAPI server MUST be running.
const API_URL = 'http://127.0.0.1:8000/query'; 

// **MUST** be exported as 'export default' for Docusaurus to read it
export default function Chatbot() {
    const [query, setQuery] = useState('');
    const [answer, setAnswer] = useState('Ask a question about the book content!');
    const [sources, setSources] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    // Advanced Requirement: Listen for selected text
    useEffect(() => {
        const handleSelection = () => {
            const selectedText = window.getSelection()?.toString().trim();
            if (selectedText) {
                // If text is selected, pre-fill the query box
                setQuery(`Question based on selected text: "${selectedText}"`);
            }
        };
        document.addEventListener('mouseup', handleSelection);
        document.addEventListener('keyup', handleSelection);
        return () => {
            document.removeEventListener('mouseup', handleSelection);
            document.removeEventListener('keyup', handleSelection);
        };
    }, []);

    const sendQuery = async () => {
        if (!query.trim()) return;

        setIsLoading(true);
        setAnswer('');
        setSources([]);

        try {
            // 1. Send the POST request to your FastAPI endpoint
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error('Server error or unauthorized access.');
            }

            const data = await response.json();
            
            // 2. Display the results
            setAnswer(data.final_answer || 'Could not find an answer.');
            setSources(data.sources || []);

        } catch (error) {
            console.error("RAG Query Failed:", error);
            setAnswer(`Error: Could not connect to RAG server or internal error. (${error.message})`);
        } finally {
            setIsLoading(false);
        }
    };

    // The component's HTML structure
    return (
        <div className={styles.chatbotContainer}>
            <h3 className={styles.title}>Integrated RAG Assistant</h3>
            <textarea
                className={styles.textarea}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about the book content, or select text on the page to analyze it..."
                rows={4}
            />
            <button className={styles.button} onClick={sendQuery} disabled={isLoading}>
                {isLoading ? 'Thinking...' : 'Get Answer'}
            </button>
            
            <div className={styles.responseArea}>
                {isLoading ? (
                    <p>Processing...</p>
                ) : (
                    <p className={styles.answerText}>{answer}</p>
                )}
                {sources.length > 0 && (
                    <div className={styles.sources}>
                        **Sources:** {sources.map(s => <span key={s} className={styles.sourceTag}>{s}</span>)}
                    </div>
                )}
            </div>
        </div>
    );
}