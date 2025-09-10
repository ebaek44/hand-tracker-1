//  options.js 
const $ = (id) => document.getElementById(id);
chrome.storage.sync.get(
  { gestureUrl:  "http://localhost:3000", gestureNs: "/gestures" },
  ({ gestureUrl, gestureNs }) => { $("url").value = gestureUrl; $("ns").value = gestureNs; }
);
$("save").onclick = () => {
  chrome.storage.sync.set(
    { gestureUrl: $("url").value.trim(), gestureNs: $("ns").value.trim() || "/gestures" },
    () => { $("status").textContent = "Saved"; setTimeout(()=>$("status").textContent="", 1000); }
  );
};
