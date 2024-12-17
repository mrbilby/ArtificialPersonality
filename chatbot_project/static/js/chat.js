/****************************************************
 * Global prevention of default drag/drop behavior
 ****************************************************/

window.addEventListener('dragover', function(e) {
    e.preventDefault();
    e.stopPropagation();
}, false);

window.addEventListener('drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
}, false);


/****************************************************
 * DOM references and utility functions
 ****************************************************/

let messageContainer = document.getElementById('messageContainer');
let messageInput = document.getElementById('messageInput');

// Store uploaded files content in memory
const uploadedFiles = {}; // { filename: "file content", ... }

function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const formattedMessage = message
        .replace(/###\s*(.*)/g, '<h3>$1</h3>')
        .replace(/(\d+\.\s+[^\n]+)/g, '<div class="list-item">$1</div>')
        .replace(/([\u{1F300}-\u{1F6FF}])/gu, '<span class="emoji">$1</span>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    content.innerHTML = formattedMessage;
    messageDiv.appendChild(content);
    messageContainer.appendChild(messageDiv);
    messageContainer.scrollTop = messageContainer.scrollHeight;
}

function toggleDiagnosticPanel() {
    const panel = document.getElementById('diagnosticPanel');
    const isVisible = localStorage.getItem('diagnosticMode') === 'true';
    panel.style.display = isVisible ? 'flex' : 'none';
}

if (document.getElementById('diagnosticMode')) {
    const checkbox = document.getElementById('diagnosticMode');
    checkbox.checked = localStorage.getItem('diagnosticMode') === 'true';
    toggleDiagnosticPanel(); 
    
    checkbox.addEventListener('change', (e) => {
        localStorage.setItem('diagnosticMode', e.target.checked);
        toggleDiagnosticPanel();
    });
}

function addDiagnosticOutput(output) {
    if (!output) return;
    if (localStorage.getItem('diagnosticMode') !== 'true') return;

    const diagnosticPanel = document.getElementById('diagnosticPanel');
    const diagnosticOutput = document.getElementById('diagnosticOutput');
    
    diagnosticPanel.style.display = 'flex';
    const timestamp = new Date().toISOString();
    const formattedOutput = `<pre style="color: #00ff00; margin: 0;">[${timestamp}]\n${output}\n\n</pre>`;
    diagnosticOutput.innerHTML += formattedOutput;
    diagnosticOutput.scrollTop = diagnosticOutput.scrollHeight;
}

// Add thinking mode event listener
document.getElementById('thinkingMode').addEventListener('change', function(e) {
    const timeContainer = document.getElementById('thinkingTimeContainer');
    timeContainer.style.display = e.target.checked ? 'block' : 'none';
});
async function sendMessage(isBye = false) {
    const message = isBye ? 'bye' : messageInput.value.trim();
    if (!message && !isBye) return;

    const thinkingModeCheckbox = document.getElementById('thinkingMode');
    const thinkingMode = thinkingModeCheckbox.checked;
    const thinkingTime = thinkingMode ? parseInt(document.getElementById('thinkingTime').value) : 0;

    if (thinkingMode && (thinkingTime < 1 || thinkingTime > 60)) {
        addMessage('Please enter a valid thinking time between 1 and 60 minutes.');
        return;
    }

    addMessage(message, true); // Add the user's message to the chat
    messageInput.value = ''; // Clear the input field

    let progressInterval;

    if (thinkingMode) {
        const overlay = document.getElementById('thinkingOverlay');
        overlay.style.display = 'block';
        const progressDiv = document.getElementById('thinkingProgress');
        let currentIteration = 0;
        let remainingTime = thinkingTime * 60;

        progressInterval = setInterval(() => {
            if (remainingTime <= 0) {
                clearInterval(progressInterval);
                progressDiv.textContent = `Thinking process completed. Waiting for final response...`;
                return;
            }

            if (remainingTime % 60 === 0) {
                currentIteration++;
                progressDiv.textContent = `Thinking iteration ${currentIteration}/${thinkingTime}... (${remainingTime} seconds remaining)`;
            } else {
                progressDiv.textContent = `Thinking iteration ${currentIteration}/${thinkingTime}... (${remainingTime} seconds remaining)`;
            }

            remainingTime--;
        }, 1000);
    }

    try {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                personality: personalityName,
                is_bye: isBye,
                diagnostic_mode: localStorage.getItem('diagnosticMode') === 'true',
                file_contents: uploadedFiles,
                thinking_mode: thinkingMode,
                thinking_minutes: thinkingTime,
            }),
        });

        const data = await response.json();

        if (data.error) {
            addMessage(`Error: ${data.error}`);
            addDiagnosticOutput(`Error: ${data.error}`); // Ensure diagnostic output is handled
        } else {
            addMessage(data.response);

            if (data.diagnostic_output) {
                console.log('Diagnostic Output:', data.diagnostic_output); // Debugging log
                addDiagnosticOutput(data.diagnostic_output); // Dynamically add output
            }

            if (data.terminated) {
                addMessage('Chat has ended. Returning to main menu...');
                setTimeout(() => {
                    window.location.href = '/';
                }, 2000);
                return;
            }

            if (data.thinking_complete) {
                if (data.diagnostic_output) {
                    addDiagnosticOutput(data.diagnostic_output);
                }
            }
        }
    } catch (error) {
        addMessage('Error: Could not send message');
        addDiagnosticOutput(`Error: ${error.message}`);
    } finally {
        if (thinkingMode) {
            clearInterval(progressInterval);
            const overlay = document.getElementById('thinkingOverlay');
            overlay.style.display = 'none';
            thinkingModeCheckbox.checked = false;
            document.getElementById('thinkingTimeContainer').style.display = 'none';
        }

        messageInput.disabled = false;
    }
}



function clearDiagnostics() {
    const diagnosticOutput = document.getElementById('diagnosticOutput');
    if (diagnosticOutput) diagnosticOutput.innerHTML = '';
}

async function endChat() {
    if (confirm('Are you sure you want to end this chat? All memories will be saved.')) {
        await sendMessage(true);
    }
}

messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});


/****************************************************
 * File handling functions
 ****************************************************/

async function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

async function handleFiles(files) {
    const fileList = document.getElementById('fileList');
    const MAX_FILE_SIZE = 100 * 1024; // 100KB

    for (const file of files) {
        if (file.size > MAX_FILE_SIZE) {
            addMessage(`Error: File ${file.name} exceeds the 100KB size limit.`, true);
            continue;
        }

        try {
            const content = await readFileContent(file);

            // Store file content in memory for later use
            uploadedFiles[file.name] = content;

            // Display the file in the UI
            const fileEntry = document.createElement('div');
            fileEntry.className = 'file-entry';
            fileEntry.innerHTML = `
                <span class="file-name" title="${file.name}">${file.name}</span>
                <button class="remove-file" onclick="this.parentElement.remove(); delete uploadedFiles['${file.name}'];">×</button>
            `;
            fileList.appendChild(fileEntry);

        } catch (error) {
            console.error('Error processing file:', error);
            addMessage(`Error processing file ${file.name}: ${error.message}`, true);
        }
    }
}


/****************************************************
 * Set up the drop zone once the DOM is loaded
 ****************************************************/

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('fileDropZone');
    const fileList = document.getElementById('fileList');

    if (!dropZone || !fileList) {
        console.error('fileDropZone or fileList not found in the DOM. Check your HTML template.');
        return;
    }

    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-active');
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = 'copy';
        dropZone.classList.add('drag-active');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-active');
    });

    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-active');

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            await handleFiles(files);
        }
    });
});