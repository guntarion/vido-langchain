import { OpenAI } from "langchain/llms/openai";
// https://js.langchain.com/docs/integrations/vectorstores/faiss#setup
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";

import express from 'express'
import http from 'http'
import { fileURLToPath } from "url";
import path, {dirname} from 'path';
import * as dotenv from 'dotenv'
dotenv.config()

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const app = express();
const port = process.env.PORT || 3000;

/* Create HTTP server */
http.createServer(app).listen(port);
console.info("listening on port " + port);

// app.listen(port, () => {
//   console.log(`Server is running on port ${port}`);
// });

/* Get endpoint to check current status  */
app.get('/api/health', async (req, res) => {
  res.json({
    success: true,
    message: 'Server is healthy',
  })
})

app.use(express.json()); // Add this line at the beginning of your file

app.post('/ask', async (req, res) => {
    try {
        const llmA = new OpenAI({ modelName: "gpt-3.5-turbo"});
        const chainA = loadQAStuffChain(llmA);
        const directory = process.env.DIR //saved directory in .env file
        
        const loadedVectorStore = await FaissStore.load(
          directory,
          new OpenAIEmbeddings()
        );
          
        const question = req.body.question; // Get the question from the request body
        const result = await loadedVectorStore.similaritySearch(question, 1);
        const resA = await chainA.call({
          input_documents: result,
          question,
        });
        // console.log({ resA });
        res.json({ result: resA }); // Send the response as JSON
    } 
      
    catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
    }
});