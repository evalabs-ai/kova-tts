/**
 * The voice-source switch's sliding bubble.
 *
 * The switch is a Gradio Radio styled as a row of pills (content.py). Gradio marks the picked
 * option by giving its label a `selected` class, so on its own the highlight would jump. Instead
 * one thumb sits behind the labels and is moved onto whichever is selected, and CSS animates the
 * move. It follows the class rather than clicks, so a source set from Python -- a finished clone
 * selects "Your recording" -- slides there too.
 *
 * Without this script the switch still works: the picked label keeps its own background, which
 * the stylesheet only hands over to the thumb once `data-sliding` is set here.
 */

(function () {
    "use strict";

    function place(wrap, thumb, animate) {
        const picked = wrap.querySelector("label.selected");
        // Nothing picked, or the switch is hidden and has no geometry to measure.
        if (!picked || !wrap.offsetWidth) {
            thumb.style.opacity = "0";
            return;
        }
        if (!animate) thumb.style.transition = "none";
        thumb.style.width = `${picked.offsetWidth}px`;
        thumb.style.height = `${picked.offsetHeight}px`;
        thumb.style.transform = `translate(${picked.offsetLeft}px, ${picked.offsetTop}px)`;
        thumb.style.opacity = "1";
        if (!animate) {
            void thumb.offsetWidth; // Commit the jump before the transition comes back.
            thumb.style.transition = "";
        }
    }

    function install(wrap) {
        const thumb = document.createElement("span");
        thumb.className = "kova-thumb";
        thumb.setAttribute("aria-hidden", "true");
        wrap.prepend(thumb);
        wrap.dataset.sliding = "";
        // The first placement, and any change of layout -- fonts arriving, the phone's two-by-two
        // grid -- jump straight there: only a change of source should be seen to move.
        place(wrap, thumb, false);
        new ResizeObserver(() => place(wrap, thumb, false)).observe(wrap);
        new MutationObserver(() => place(wrap, thumb, true)).observe(wrap, {
            subtree: true,
            attributes: true,
            attributeFilter: ["class"],
        });
    }

    // Gradio mounts the page after the load handler starts, so wait for the switch to exist.
    const ready = setInterval(() => {
        // The label's parent, not ".wrap": Gradio's status tracker is a .wrap as well.
        const wrap = document.querySelector("#kova-source label")?.parentElement;
        if (!wrap) return;
        clearInterval(ready);
        install(wrap);
    }, 50);
})();
