# Instructions

This is a repo full of video graphics


1. Find an interesting physics/math/deep learning/data science problem that requires heavier computation to render. For example, visualizing the social network of a texting dataset. Be creative. Remember, you aren't as constrained by time, space. For example, you can visualize frequency spectrums that humans can't see or condense billions of years within a seconds. You may download any dataset our r2 experiments bucket under a folder named `datasets`
2. Reference the knowledge folder. Add any redundant scripts into `scripts`. 
3. Render the phenomenon into 720p mp4 no longer than 30 seconds and drop any intermediate files in the drafts folder under a slug. Add a brief readme on how to manually recreate the results.
4. Use observablenotebook kit to render the synopsis into a elegant and simple html file. Include the synopsis and the embeded video and the mathematical background of the phenomenon to understand what you're seeing. Put all intermediate files used to render all of this in a slug folder in the drafts folder.
<<<<<<< HEAD
5. Compile your video and html file in the folder `dist` with a unique slug. Screenshot a section of the video that reasonable covers the topic. Upload the files to r2's `experiments` bucket under a folder named `graphics` with the same slug. The file should be named index.html.
6. Merge into main, no need for PR or branches
7. Use the digital ocean to add to the table `graphics` in the database `machinedata` with the following columns: `slug`, `title`, `description`, `r2_link`, `created_at`, `video_link`, `image_link`. r2_link should be the public facing html link that we uploaded.
8. In the pgvector column of the table `graphics` in the database `machinedata`, add a vector representation of the story. Use the `text-embedding-3-large` model to generate the vector representation (3072 dims). The vector should be stored in the column `vector`. Reach `text-embedding-3-large` through OpenRouter using the modal secret `openrouter-secret` (`OPENROUTER_API_KEY`); POST to `https://openrouter.ai/api/v1/embeddings` with model `openai/text-embedding-3-large`.
=======
5. Compile your video and html file in the folder `dist` with a unique slug that way the github repo doesn't explode in memory size. Screenshot a section of the video that reasonable covers the topic. Upload the files to r2's `experiments` bucket under a folder named `graphics` with the same slug. The file should be named index.html.
6. Use the digital ocean to add to the table `graphics` in the database `machinedata` with the following columns: `slug`, `title`, `description`, `r2_link`, `created_at`, `video_link`, `image_link`. r2_link should be the public facing html link that we uploaded.
7. In the pgvector column of the table `graphics` in the database `machinedata`, add a vector representation of the story. Use the `text-embedding-3-large` model to generate the vector representation (3072 dims). The vector should be stored in the column `vector`. Reach `text-embedding-3-large` through OpenRouter using the modal secret `openrouter-secret` (`OPENROUTER_API_KEY`); POST to `https://openrouter.ai/api/v1/embeddings` with model `openai/text-embedding-3-large`.
>>>>>>> dc98f26 (ok)

Tools: Use modal when necessary. Use the web as however necessary but research in depth to make the story vivid and realistic. Use r2 (modal secrets called `r2`) and github (called `github`) and digitalocean (called `do_db`) and openrouter (called `openrouter-secret`) and kaggle cli (called `kaggle`)

Infra notes (discovered):
- `r2` secret exposes `ENDPOINT`, `ACCESS_KEY_ID`, `SECRET_KEY_ID` (S3 API only — no Cloudflare account API). Upload with the S3 API (e.g. boto3, `region_name="auto"`). Public base URL for the `experiments` bucket is `https://experiments.funnyscar.com`, so `r2_link` = `https://experiments.funnyscar.com/fanfiction/<slug>/index.html`.
- `do_db` secret exposes `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `DATABASE` (=`defaultdb`). Connect with `sslmode=require`. The `fanfiction` database and its `vector` extension (pgvector 0.8.6) have been created; `vector` column is `vector(3072)`.

If you need special permissions or there is ambiguity you can't resolve, move on to a new story.
