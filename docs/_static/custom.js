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

        // Remove the tooltip attribute to prevent the description from popping up
        el.removeAttribute('tooltip');
    });
});