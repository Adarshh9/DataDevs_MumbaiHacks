import React, { useState } from 'react';
import axios from 'axios';

const CombinedComponent = () => {
  const [selectedOption, setSelectedOption] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [linkInput, setLinkInput] = useState('');
  const [showWarning, setShowWarning] = useState(false);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
    setShowWarning(false); // Reset warning when dropdown value changes
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleLinkInputChange = (event) => {
    setLinkInput(event.target.value);
  };

  const handleSubmit = async () => {
    if (selectedOption === '') {
      setShowWarning(true); // Show warning if dropdown option is not selected
      return; // Prevent further execution
    }

    try {
      // Handling option submission
      const optionResponse = await axios.post(`http://127.0.0.1:5000/store_option/${selectedOption}`);
      if (optionResponse.status === 200) {
        console.log('Option sent successfully!');
      }

      // Handling file upload
      const formData = new FormData();
      formData.append('file', selectedFile);

      const fileResponse = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (fileResponse.status === 200) {
        console.log('File uploaded successfully!');
      }

      // Handling link input submission
      const linkResponse = await axios.post(`http://127.0.0.1:5000/store_link/${linkInput}`);
      if (linkResponse.status === 200) {
        console.log('Link input sent successfully!');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h3>Combined Component</h3>
      <input type="file" onChange={handleFileChange} />
      <br />
      <select value={selectedOption} onChange={handleOptionChange}>
        <option value="">Select an option</option>
        <option value="AI">AI</option>
        <option value="Web Development">Web Development</option>
        <option value="Option3">Option 3</option>
        {/* Add more options as needed */}
      </select>
      {showWarning && <p style={{ color: 'red' }}>Please select an option!</p>}
      <br />
      <input type="text" value={linkInput} onChange={handleLinkInputChange} placeholder="Enter link URL" />
      <br />
      <button onClick={handleSubmit}>Submit All</button>
    </div>
  );
};

export default CombinedComponent;
