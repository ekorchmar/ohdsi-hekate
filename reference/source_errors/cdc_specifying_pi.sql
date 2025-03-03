select
    c1.concept_id,
    c1.concept_name,
    r.relationship_id,
    c2.concept_id as ingredient_concept_id,
    c2.concept_name as ingredient_concept_name,
    c2.concept_class_id as ingredient_concept_class_id
from concept as c1
inner join
    concept_relationship as r
    on
        c1.concept_id = r.concept_id_1
        and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        and r.invalid_reason is null
        and c1.invalid_reason is null
inner join
    concept as c2
    on
        c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        and c2.invalid_reason is null
        and r.concept_id_2 = c2.concept_id
where
    c1.concept_class_id = 'Clinical Drug Form'
    and c2.concept_class_id = 'Precise Ingredient'
    and r.relationship_id = 'RxNorm has ing';
