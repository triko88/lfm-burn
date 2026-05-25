use tokenizers::Tokenizer;

#[test]
fn tokenizer_fixture_loads() {
    let tok = Tokenizer::from_file("test_repo/tokenizer.json")
        .expect("tokenizer.json should load");
    assert_eq!(tok.get_vocab_size(true), 8);
}

#[test]
fn tokenizer_fixture_decodes_regular_tokens() {
    let tok = Tokenizer::from_file("test_repo/tokenizer.json")
        .expect("tokenizer.json should load");
    let ids: Vec<u32> = vec![4, 5, 6, 6, 7];
    let decoded = tok.decode(&ids, false).expect("decode should succeed");
    assert_eq!(decoded, "abccd", "expected 'abccd', got {decoded:?}");
}

#[test]
fn tokenizer_fixture_decodes_with_special_tokens() {
    let tok = Tokenizer::from_file("test_repo/tokenizer.json")
        .expect("tokenizer.json should load");
    let ids: Vec<u32> = vec![2, 4, 5, 3];
    let decoded = tok.decode(&ids, false).expect("decode should succeed");
    assert_eq!(decoded, "[BOS]ab[EOS]", "expected '[BOS]ab[EOS]', got {decoded:?}");
}

#[test]
fn tokenizer_fixture_id_conversion() {
    let tok = Tokenizer::from_file("test_repo/tokenizer.json")
        .expect("tokenizer.json should load");
    assert_eq!(tok.token_to_id("a"), Some(4));
    assert_eq!(tok.token_to_id("d"), Some(7));
    assert_eq!(tok.token_to_id("[PAD]"), Some(0));
    assert_eq!(tok.token_to_id("[BOS]"), Some(2));
    assert_eq!(tok.token_to_id("[EOS]"), Some(3));
    assert_eq!(tok.token_to_id("[UNK]"), Some(1));
    assert_eq!(tok.id_to_token(4), Some("a".into()));
    assert_eq!(tok.id_to_token(0), Some("[PAD]".into()));
    assert_eq!(tok.id_to_token(8), None);
}
