import webview
import torch
import numpy as np
from PIL import Image
import base64
import io
from engine import get_model


model = get_model('aetheris_model.pth')

html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { 
            background: #0a0a0c; color: #8a8a95; font-family: 'Monaco', monospace; 
            margin: 0; display: flex; flex-direction: column; height: 100vh;
        }
        .header { 
            padding: 10px; background: #111116; border-bottom: 1px solid #1f1f2e; 
            text-align: center; color: #00f2ff; font-size: 11px; letter-spacing: 2px;
        }
        #chat { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 8px; }
        .msg { max-width: 85%; padding: 8px 12px; border-radius: 4px; font-size: 12px; border-left: 2px solid #1f1f2e; }
        .ai { background: #111116; color: #00f2ff; }
        .user { background: #16161d; align-self: flex-end; border-left: 0; border-right: 2px solid #333; }
        
        #draw-zone {
            background: #000; width: 200px; height: 200px; margin: 10px auto;
            border: 1px solid #00f2ff33; cursor: crosshair; touch-action: none;
        }
        .footer { padding: 15px; background: #0d0d12; border-top: 1px solid #1f1f2e; display: flex; gap: 8px; }
        button {
            flex: 1; background: #111; border: 1px solid #333; color: #666; 
            padding: 10px; cursor: pointer; font-size: 9px; text-transform: uppercase;
        }
        button:hover { border-color: #00f2ff; color: #00f2ff; }
    </style>
</head>
<body>
    <div class="header">AETHERIS NEURAL CORE v2.1</div>
    <div id="chat"><div class="msg ai">SYSTEM: Draw a digit (0-9) below and press GUESS.</div></div>
    
    <canvas id="draw-zone" width="200" height="200"></canvas>

    <div class="footer">
        <button onclick="clearCanvas()">Clear</button>
        <button onclick="sendImage()" style="border-color: #00f2ff66; color: #00f2ff;">Guess</button>
    </div>

    <script>
        const canvas = document.getElementById('draw-zone');
        const ctx = canvas.getContext('2d');
        const chat = document.getElementById('chat');
        let drawing = false;

        ctx.strokeStyle = "white";
        ctx.lineWidth = 15;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";

        canvas.onmousedown = () => drawing = true;
        canvas.onmouseup = () => { drawing = false; ctx.beginPath(); };
        canvas.onmousemove = (e) => {
            if(!drawing) return;
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
        };

        function addMsg(text, type) {
            const d = document.createElement('div');
            d.className = 'msg ' + type;
            d.innerText = text;
            chat.appendChild(d);
            chat.scrollTop = chat.scrollHeight;
        }

        function clearCanvas() {
            ctx.fillStyle = "black";
            ctx.fillRect(0,0,200,200);
            addMsg("Canvas cleared.", "ai");
        }

        function sendImage() {
            const dataUrl = canvas.toDataURL('image/png');
            addMsg("Analyzing input...", "user");
            pywebview.api.predict(dataUrl).then(res => {
                addMsg("RESULT: " + res, "ai");
            });
        }
        clearCanvas();
    </script>
</body>
</html>
"""

class Bridge:
    def predict(self, data_url):
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(data)).convert('L')  
        img = img.resize((28, 28))  
        
        img_array = np.array(img) / 255.0
        img_array = (img_array - 0.1307) / 0.3081
        tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output).item()
            conf = torch.nn.functional.softmax(output, dim=1).max().item() * 100
        
        return f"Digit {pred} ({conf:.1f}%)"

api = Bridge()
webview.create_window('Aetheris Interface', html=html_content, js_api=api, width=400, height=600)
webview.start()
