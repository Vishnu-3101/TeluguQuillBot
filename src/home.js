import React, { useState } from "react";
import axios from "axios";
import "./home.css";

const Home = () => {
  const [newMessage, setNewMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [showMenu, setShowMenu] = useState(false); // State variable to track menu visibility
  const [showProfileDropdown, setShowProfileDropdown] = useState(false); // State variable to track profile dropdown visibility

  const send = () => {
    const botMessage = { id: messages.length, text: newMessage, primary: true };
    setMessages([...messages, botMessage]);
    setNewMessage("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: newMessage,
      });
      const data = await res.json(); // Extract response text
      // Process the response as needed
      console.log(data.output);
      const botMessage = {
        id: messages.length,
        text: newMessage,
        primary: true,
      };
      const responseMessage = {
        id: messages.length + 1,
        text: data.output,
        primary: false,
      };
      setMessages([...messages, botMessage, responseMessage]); // Update state with both user's message and bot's response
      setNewMessage(""); // Clear the input field
    } catch (error) {
      console.log(error);
    }
  };

  // Function to toggle the side menu
  const toggleMenu = () => {
    setShowMenu(!showMenu);
  };

  // Function to toggle the profile dropdown
  const toggleProfileDropdown = () => {
    setShowProfileDropdown(!showProfileDropdown);
  };

  return (
    <div id="app" className="chat-app">
      <button className="menu-toggle" onClick={toggleMenu}>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          fill="currentColor"
          class="bi bi-list"
          viewBox="0 0 16 16"
        >
          <path
            fill-rule="evenodd"
            d="M2.5 12a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5"
          />
        </svg>
      </button>
      <div className={`side-menu ${showMenu ? "open" : ""}`}>
        <div className="side-menu__header">History</div>
        <div className="side-menu__content">
          {/* Side menu content goes here */}
        </div>
        <div className="side-menu__footer">
          <button className="profile-button" onClick={toggleProfileDropdown}>
            Profile
          </button>
          {showProfileDropdown && (
            <div className="dropdown">
              {/* Dropdown content goes here */}
              <ul>
                <li>Settings</li>
                <li>Log Out</li>
              </ul>
            </div>
          )}
        </div>
      </div>
      <div className="chatbox">
        <div className="chatbox__messages">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`chatbox__messageBox ${
                message.primary ? "chatbox__messageBox--primary" : ""
              }`}
            >
              <div
                className={`chatbox__message ${
                  message.primary ? "chatbox__message--primary" : ""
                }`}
              >
                {message.text}
              </div>
            </div>
          ))}
        </div>

        <div className="chatbox__inputPanel">
          <form onSubmit={handleSubmit}>
            <input
              className="chatbox__input"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyUp={(e) => e.key === "Enter" && send()}
              placeholder="Ask me anything"
            />
          </form>
        </div>
      </div>
    </div>
  );
};

export default Home;
