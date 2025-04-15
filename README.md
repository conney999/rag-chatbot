```mermaid
graph LR
    subgraph Data Ingestion & Indexing
        A[Data Sources (e.g., Text Files, PDFs)] --> B(Document Loading - LangChain);
        B --> C(Text Splitting - LangChain);
        C --> D(Embedding Generation - Sentence Transformers);
        D --> E(Vector Store - ChromaDB);
    end

    subgraph Retrieval & Generation
        F[User Query] --> G(Embedding Generation - Sentence Transformers);
        G --> H(Semantic Search - ChromaDB);
        H --> I(Retrieve Relevant Documents - LangChain);
        I --> J(Prompt Construction - LangChain);
        J --> K(Language Model - Hugging Face Transformers (via LangChain));
        K --> L[Generated Answer];
    end

    style Data Ingestion & Indexing fill:#f9f,stroke:#333,stroke-width:2px
    style Retrieval & Generation fill:#ccf,stroke:#333,stroke-width:2px

    E -- Stores Embeddings --> H
    I -- Provides Context --> K
