import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

function Login() {
  const [isSignInActive, setSignInActive] = useState(true);
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const history = useNavigate();

  const handleToggle = () => {
    setSignInActive(!isSignInActive);
  };


  const handleSignup = async (formData) => {
    try {
      // Remove the "id" field from formData
      const { id, ...userData } = formData;
  
      const response = await fetch('http://localhost:5000/Register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData) // Send userData to the /register endpoint without the "id" field
      });
      const data = await response.json();
      console.log(data);
  
      if (response.ok) {
        // If registration is successful, redirect to the home page
        alert("Continue with Sign in");
      } else {
        // If registration fails, display an error message
        console.error('Error signing up:', data.error);
      }
    } catch (error) {
      console.error('Error signing up:', error);
    }
  }


  const handleSignin = async (formData) => {
    try {
      // Remove the "id , username " field from formData
      
      const { id, username , ...userData } = formData;
  
      const response = await fetch('http://localhost:5000/Signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData) // Send userData to the /register endpoint without the "id" field
      });
      const data = await response.json();
      console.log(data);
  
      if (response.ok) {
        // If registration is successful, redirect to the home page
        history('/home');
      } else {
        // If registration fails, display an error message
        console.error('Error signing in:', data.error);
      }
    } catch (error) {
      console.error('Error signing in:', error);
    }
    
  }
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };
  const handleSubmit = async (e) => {
    const children =  e.target.children;
    const button = children[children.length - 1];
    const dec = button.textContent;
    console.log(button.textContent);

    e.preventDefault();

    try {
      if  (dec === 'Sign In') {
        await handleSignin(formData)
      } else{
        await handleSignup(formData)
      }
    } catch (error) {
      if  (dec === 'Sign In') {
        console.error('Error signing in:', error);
      } else{
        console.error('Error signing up:', error);
      }
      
    }

    
  };
  

  return (
    <div className={`container ${isSignInActive ? '' : 'active'}`}>
      <div className={`form-container ${isSignInActive ? 'sign-in' : 'sign-up'}`}>
        <form onSubmit={handleSubmit}>
          <h1>{isSignInActive ? 'Sign In' : 'Create Account'}</h1>
          <div className="social-icons">
            {/* Your social icons */}
          </div>
          {isSignInActive ? (
            <span>or use your email password</span>
          ) : (
            <span>or use your email for registration</span>
          )}
          {!isSignInActive && <input type="text" name="username" placeholder="Name" value={formData.name} onChange={handleChange} />}
          <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} />
          <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} />
          {isSignInActive && <a href="#">Forget Your Password?</a>}
          <button type="submit">{isSignInActive ? 'Sign In' : 'Sign Up'}</button>
        </form>
      </div>
      <div className="toggle-container">
        <div className="toggle">
          <div className={`toggle-panel toggle-left ${isSignInActive ? 'active' : ''}`}>
            <h1>Welcome Back!</h1>
            <p>Enter your personal details to use all site features</p>
            <button className="hidden" onClick={handleToggle}>Sign In</button>
          </div>
          <div className={`toggle-panel toggle-right ${!isSignInActive ? 'active' : ''}`}>
            <h1>Hello, Friend!</h1>
            <p>Register with your personal details to use all site features</p>
            <button className="hidden" onClick={handleToggle}>Sign Up</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
