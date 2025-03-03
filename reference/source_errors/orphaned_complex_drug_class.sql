-- Complex Drug Classes with no ingredient ancestor
select
    cd.concept_class_id,
    count(*) as cnt
from concept as cd
where
    cd.domain_id = 'Drug'
    and cd.standard_concept = 'S'
    and cd.concept_class_id != 'Ingredient'
    and cd.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    and not exists (
        select 1
        from concept_ancestor as a
        inner join concept as ci
            on
                a.descendant_concept_id = cd.concept_id
                and a.ancestor_concept_id = ci.concept_id
                and ci.concept_class_id = 'Ingredient'
                and ci.domain_id = 'Drug'
                and ci.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    )
group by cd.concept_class_id
