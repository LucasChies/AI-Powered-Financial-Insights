document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("search");
    const topics = document.querySelectorAll(".topic");

    // Filter topics based on search input
    searchInput.addEventListener("input", (event) => {
        const query = event.target.value.toLowerCase();
        topics.forEach((topic) => {
            const title = topic.querySelector(".topic-title").textContent.toLowerCase();
            if (title.includes(query)) {
                topic.style.display = "block";
            } else {
                topic.style.display = "none";
            }
        });
    });

    // Collapsible sections
    topics.forEach((topic) => {
        const title = topic.querySelector(".topic-title");
        title.addEventListener("click", () => {
            const content = topic.querySelector(".topic-content");
            content.style.display = content.style.display === "none" ? "block" : "none";
        });
        topic.querySelector(".topic-content").style.display = "none"; // Start collapsed
    });
});
