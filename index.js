import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import fs from "fs";
import { parse } from "csv-parse/sync";

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

// A helper function that breaks an array into chunks of size batchSize
const chunks = (array, batchSize = 200) => {
  const chunks = [];

  for (let i = 0; i < array.length; i += batchSize) {
    chunks.push(array.slice(i, i + batchSize));
  }

  return chunks;
};

async function generateEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  
  // The OpenAI embeddings are 1536 dimensions, but our Pinecone index is 1024
  // We'll take the first 1024 dimensions as a simple approach to match dimensions
  return response.data[0].embedding.slice(0, 1024);
}

async function upsertRecords(indexName, records) {
  const index = pinecone.index(indexName);
  
  // Get index stats to verify dimensions
  const indexStats = await index.describeIndexStats();
  const indexDimension = indexStats.dimension;
  console.log(`Pinecone index '${indexName}' has dimension: ${indexDimension}`);
  
  // First generate all embeddings for records
  console.log(`Generating embeddings for ${records.length} records...`);
  const recordsWithEmbeddings = await Promise.all(
    records.map(async (record) => {
      const textForEmbedding = `${record.Name}: ${record.Description} Category: ${record.Category} Integration: ${record.Integration}`;
      
      let embedding = await generateEmbedding(textForEmbedding);
      
      // Double-check dimension
      if (embedding.length !== indexDimension) {
        console.log(`Warning: embedding dimension (${embedding.length}) doesn't match index dimension (${indexDimension})`);
        // Resize embedding to match index dimension
        if (embedding.length > indexDimension) {
          embedding = embedding.slice(0, indexDimension);
        } else {
          // Pad with zeros if embedding is smaller than index dimension (unlikely case)
          embedding = [...embedding, ...Array(indexDimension - embedding.length).fill(0)];
        }
      }
      
      return {
        id: record.Name.toLowerCase().replace(/\s+/g, '-'),
        values: embedding,
        metadata: {
          name: record.Name,
          category: record.Category,
          integration: record.Integration,
          description: record.Description,
          url: record.Url,
          selectableWith: record["Selectable with"] || "",
        },
      };
    })
  );
  
  // Split records into chunks to avoid exceeding the 4MB request limit
  const recordChunks = chunks(recordsWithEmbeddings, 50); // Adjust batch size as needed
  console.log(`Splitting upsert into ${recordChunks.length} batches...`);
  
  // Process each chunk
  let totalUpserted = 0;
  for (let i = 0; i < recordChunks.length; i++) {
    const chunk = recordChunks[i];
    console.log(`Upserting batch ${i + 1}/${recordChunks.length} (${chunk.length} records)...`);
    
    // Verify dimensions of first vector in chunk (for debugging)
    if (chunk.length > 0) {
      console.log(`First vector in batch ${i + 1} has dimension: ${chunk[0].values.length}`);
    }
    
    try {
      const upsertResponse = await index.upsert(chunk);
      totalUpserted += chunk.length;
      console.log(`Batch ${i + 1} completed successfully, upserted ${chunk.length} records`);
    } catch (error) {
      console.error(`Error in batch ${i + 1}:`, error.message);
      
      // If a chunk is still too large, try with even smaller chunks
      if (chunk.length > 10 && error.message.includes("message length too large")) {
        console.log("Batch too large, retrying with smaller chunks...");
        // Create smaller sub-chunks and process them
        const smallerChunks = chunks(chunk, Math.floor(chunk.length / 2));
        for (const smallerChunk of smallerChunks) {
          try {
            console.log(`Upserting smaller chunk with ${smallerChunk.length} records...`);
            const response = await index.upsert(smallerChunk);
            totalUpserted += smallerChunk.length;
            console.log(`Smaller chunk completed, upserted ${smallerChunk.length} records`);
          } catch (smallerError) {
            console.error(`Failed to upsert smaller chunk:`, smallerError.message);
          }
        }
      } else {
        console.error(`Failed to upsert batch ${i + 1}. Continuing with next batch.`);
      }
    }
    
    // Add a small delay between batches to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  return { upsertedCount: totalUpserted };
}

async function getRecommendations(indexName, metricName, filters = {}, topK = 5) {
  const index = pinecone.index(indexName);
  
  // Find the vector for the query metric
  const metricToQuery = await index.fetch([metricName.toLowerCase().replace(/\s+/g, '-')]);
  
  if (!metricToQuery.records || Object.keys(metricToQuery.records).length === 0) {
    throw new Error(`Metric "${metricName}" not found in the database`);
  }
  
  const queryVector = metricToQuery.records[metricName.toLowerCase().replace(/\s+/g, '-')].values;
  
  // Prepare filter based on user criteria
  const filterObj = {};
  for (const [key, value] of Object.entries(filters)) {
    if (value) filterObj[key] = { $eq: value };
  }
  
  // Query for similar metrics
  const queryResponse = await index.query({
    vector: queryVector,
    topK,
    filter: Object.keys(filterObj).length > 0 ? filterObj : undefined,
    includeMetadata: true,
  });
  
  return queryResponse.matches;
}

async function loadCSVData(filePath) {
  const csvData = fs.readFileSync(filePath, 'utf8');
  const records = parse(csvData, {
    columns: true,
    skip_empty_lines: true,
  });
  return records;
}

async function createMetricsDatabase() {
  try {
    const indexName = "fields";
    
    console.log("Loading CSV data...");
    const csvPath = "./fields.csv";
    const metrics = await loadCSVData(csvPath);
    
    console.log(`Processing ${metrics.length} metrics in batches...`);
    const result = await upsertRecords(indexName, metrics);
    console.log(`Upsert completed: ${result.upsertedCount}/${metrics.length} records inserted`);
    
    // Example recommendation query
    console.log("\nGetting recommendations for 'cpc'...");
    const recommendations = await getRecommendations(indexName, "cpc", { 
      integration: "facebook-ads" 
    });
    
    console.log("Recommended metrics:");
    recommendations.forEach(match => {
      console.log(`- ${match.metadata.name} (${match.score.toFixed(2)}): ${match.metadata.description}`);
    });
    
  } catch (error) {
    console.error("Error in createMetricsDatabase:", error.message);
    if (error.stack) {
      console.error("Stack trace:", error.stack);
    }
  }
}

// Execute the main function
createMetricsDatabase();

export { generateEmbedding, upsertRecords, getRecommendations, loadCSVData, createMetricsDatabase }

