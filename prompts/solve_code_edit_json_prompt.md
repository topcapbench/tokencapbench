You are editing files in a programming exercise. Return only JSON.

Use this exact schema:

{
  "files": [
    {
      "path": "relative/path/to/file.ext",
      "content": "complete replacement content"
    }
  ]
}

Include complete file contents for every modified source file. Do not include markdown. Do not modify tests.
