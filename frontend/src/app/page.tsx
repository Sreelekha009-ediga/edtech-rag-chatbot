"use client";

import { useState } from "react";
import { useChat } from "ai/react";
import { Button, Card, CardContent, CardHeader, Divider, Typography, Box, TextField, IconButton } from '@mui/material';
import { Send, Upload, AttachFile } from '@mui/icons-material';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat",  // Vercel AI SDK proxy to Groq
  });

  const handleUpload = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);

    const endpoint = file.type.startsWith("image/") ? "/api/ingest-image" : "/api/ingest-pdf";
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        alert(`File ingested! ${data.message || data.extracted_text}`);
      } else {
        alert("Upload failed – check backend logs");
      }
    } catch (error) {
      alert("Upload error: " + error);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 2, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h3" gutterBottom align="center">
        EdTech Textbook Reader
      </Typography>
      <Typography variant="body1" gutterBottom align="center" color="textSecondary">
        Upload PDF/image → Ask questions → Get grounded answers
      </Typography>

      {/* File Upload */}
      <Box sx={{ mb: 4, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
        <Typography variant="h6" gutterBottom>
          Upload Document
        </Typography>
        <input
          type="file"
          accept="image/*,.pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          style={{ marginBottom: 8 }}
        />
        <Button
          variant="contained"
          startIcon={<Upload />}
          onClick={handleUpload}
          disabled={!file}
        >
          Ingest File
        </Button>
      </Box>

      {/* Chat Area */}
      <Box sx={{ flex: 1, overflow: 'auto', mb: 2, border: '1px solid grey.300', borderRadius: 1, p: 2 }}>
        {messages.map((m) => (
          <Box key={m.id} sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="textSecondary">
              {m.role === "user" ? "You" : "Assistant"}
            </Typography>
            <Typography>{m.content}</Typography>
            {m.sources && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption">Sources:</Typography>
                {m.sources.map((src, i) => (
                  <Card key={i} sx={{ mt: 1, p: 1 }}>
                    <Typography variant="caption">
                      {src.source} ({src.type})
                    </Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8em' }}>
                      {src.text}
                    </Typography>
                  </Card>
                ))}
              </Box>
            )}
          </Box>
        ))}
      </Box>

      {/* Input */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask a question about your documents..."
          value={input}
          onChange={handleInputChange}
          onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
          disabled={isLoading}
        />
        <IconButton
          color="primary"
          onClick={handleSubmit}
          disabled={isLoading}
        >
          <Send />
        </IconButton>
      </Box>
    </Box>
  );
}