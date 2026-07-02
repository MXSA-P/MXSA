document.addEventListener('DOMContentLoaded', () => {
    "use strict";

    // Ensure HTTP Basic Auth credentials are sent with all API fetch requests
    const _originalFetch = window.fetch;
    window.fetch = function(url, options = {}) {
        options = options || {};
        if (!options.credentials) {
            options.credentials = 'include';
        }
        return _originalFetch.call(this, url, options);
    };

    const buttons = document.querySelectorAll('.diag-btn');
    const terminal = document.getElementById('terminal-output');

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

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
            
            addTerminalLine(`Initiating test sequence: <span style="color:#fff">${escapeHtml(testName)}</span>...`, 'info');

            try {
                const response = await fetch('/api/diagnostics/run', {
                    method: 'POST',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ test: testName })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                const resultMsg = data.status || data.message || data.output || `Test '${escapeHtml(testName)}' completed successfully.`;
                addTerminalLine(escapeHtml(resultMsg), 'success');

            } catch (error) {
                addTerminalLine(`Error executing test: ${escapeHtml(error.message)}`, 'error');
            } finally {
                setTimeout(() => {
                    button.classList.remove('active');
                }, 2000);
            }
        });
    });
});
