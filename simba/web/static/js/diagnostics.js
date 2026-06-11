document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.diag-btn');
    const terminal = document.getElementById('terminal-output');

    function addTerminalLine(text, type = 'info') {
        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;
        
        const time = new Date().toLocaleTimeString([], { hour12: false });
        
        line.innerHTML = `<span class="prompt">[${time}] root@mxsa:~#</span> ${text}`;
        terminal.appendChild(line);
        
        // Auto scroll to bottom
        terminal.scrollTop = terminal.scrollHeight;
    }

    buttons.forEach(button => {
        button.addEventListener('click', async () => {
            const testName = button.getAttribute('data-test');
            
            // UI Update
            buttons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            addTerminalLine(`Initiating test sequence: <span style="color:#fff">${testName}</span>...`, 'info');

            try {
                const response = await fetch('/api/diagnostics/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ test: testName })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // Assuming the API returns a message or output in the JSON
                const resultMsg = data.message || data.output || `Test '${testName}' completed successfully.`;
                addTerminalLine(resultMsg, 'success');

            } catch (error) {
                addTerminalLine(`Error executing test: ${error.message}`, 'error');
            } finally {
                setTimeout(() => {
                    button.classList.remove('active');
                }, 2000);
            }
        });
    });
});
