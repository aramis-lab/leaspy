document.addEventListener("DOMContentLoaded", function() {
    // Find all gallery thumbnails
    document.querySelectorAll('.sphx-glr-thumbcontainer').forEach(function(el) {
        const pyfileSpan = el.querySelector('.xref.std.std-ref');
        if (pyfileSpan) {
            let pyfile = pyfileSpan.textContent.trim();
            
            // Remove the sphx_glr_auto_examples_ prefix
            pyfile = pyfile.replace(/^sphx_glr_auto_examples_/, '');
            
            // Replace .py with .html
            const htmlFile = pyfile.replace(/\.py$/, '.html');

            // Attach click event
            el.addEventListener('click', function() {
                window.location.href = htmlFile;
            });

            // Change cursor to pointer
            el.style.cursor = 'pointer';
        }
    });

    // Make checklist checkboxes interactive
    document.querySelectorAll('input.task-list-item-checkbox').forEach((cb, i) => {
        const key = `checklist-${location.pathname}-${i}`;
        cb.disabled = false;
        cb.checked = localStorage.getItem(key) === 'true';
        cb.onchange = () => localStorage.setItem(key, cb.checked);
    });
});