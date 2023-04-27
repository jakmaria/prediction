import fetch from 'node-fetch';
import express from 'express';
import bodyParser from 'body-parser';

const app = express();

app.use(bodyParser.json());

app.post('/predict', async (req, res) => {
  const input_data = req.body;
  const python_server_url = 'http://127.0.0.1:5000/predict';

  try {
    console.log('try beginning');
    const response = await fetch(python_server_url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input_data),
    });

    const prediction_data = await response.json();
    res.json(prediction_data);
  } catch (error) {
    console.error('Error fetching data from Python server:', error);
    res.status(500).json({ error: 'Error fetching data from Python server' });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
