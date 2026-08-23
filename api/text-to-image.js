import { InferenceClient } from "@huggingface/inference";

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

export default async function handler(req, res) {
  cors(res);

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.HF_TOKEN) {
    return res.status(500).json({
      error: "HF_TOKEN is missing in Vercel."
    });
  }

  try {
    const body = req.body || {};
    const prompt = typeof body.prompt === "string"
      ? body.prompt.trim()
      : "";

    const negative = typeof body.negative_prompt === "string"
      ? body.negative_prompt.trim()
      : "";

    const width = Math.min(1024, Math.max(256, Number(body.width) || 512));
    const height = Math.min(1024, Math.max(256, Number(body.height) || 512));

    if (!prompt) {
      return res.status(400).json({
        error: "Prompt is required."
      });
    }

    const client = new InferenceClient(process.env.HF_TOKEN);

    const inputs = negative
      ? `${prompt}\n\nNegative prompt: ${negative}`
      : prompt;

    const image = await client.textToImage({
      model: "black-forest-labs/FLUX.1-schnell",
      provider: "auto",
      inputs,
      parameters: {
        width,
        height
      }
    });

    const buffer = Buffer.from(await image.arrayBuffer());

    res.setHeader(
      "Content-Type",
      image.type || "image/png"
    );

    res.setHeader("Cache-Control", "no-store");

    return res.status(200).send(buffer);

  } catch (err) {
    console.error(err);

    return res.status(502).json({
      error: "Hugging Face generation failed.",
      detail: err?.message || String(err)
    });
  }
}
