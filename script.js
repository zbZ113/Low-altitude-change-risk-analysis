document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  const navToggle = document.querySelector(".nav-toggle");
  const nav = document.querySelector(".site-nav");

  navToggle?.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "关闭导航" : "打开导航");
  });

  nav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("is-open");
      navToggle?.setAttribute("aria-expanded", "false");
    });
  });

  const dialog = document.querySelector(".image-dialog");
  const dialogImage = dialog?.querySelector("img");
  const closeButton = dialog?.querySelector(".dialog-close");

  const openImage = (trigger) => {
    if (!dialog || !dialogImage) return;
    dialogImage.src = trigger.dataset.full;
    const sourceImage = trigger.querySelector("img");
    dialogImage.alt = sourceImage?.alt || "项目图片预览";
    dialog.showModal();
  };

  document.querySelectorAll(".image-openable").forEach((trigger) => {
    trigger.addEventListener("click", () => openImage(trigger));
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openImage(trigger);
      }
    });
  });

  closeButton?.addEventListener("click", () => dialog.close());
  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog?.addEventListener("close", () => {
    if (dialogImage) dialogImage.src = "";
  });
});
