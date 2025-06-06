const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");

const PORT = process.env.PORT || 5000;
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

const app = express();
app.use(bodyParser.json());

app.post("/api/compare", async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_BACKEND_URL}/compare`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with Python API:", error.message);
    res.status(500).json({ error: "Python service failed", details: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Node.js API running on port ${PORT}`);
});
