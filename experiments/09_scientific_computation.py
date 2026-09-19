from materials_rag.ingestion.scientific_computation import build_structured_data_catalog

for dataset_id, descriptor in build_structured_data_catalog().items():
    print(dataset_id, sorted(descriptor.supported_operations))
