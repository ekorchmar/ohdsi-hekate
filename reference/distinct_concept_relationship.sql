-- List of ALL relationships between RxNorm and RxNorm extension
select distinct r.relationship_id
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
        c2.invalid_reason is null
        and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        and r.concept_id_2 = c2.concept_id
